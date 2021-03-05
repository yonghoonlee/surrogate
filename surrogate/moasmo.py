#!/usr/bin/env python
# coding: utf-8
# Copyright (c) 2021 Yong Hoon Lee

import numpy as np
from copy import deepcopy
from smt.surrogate_models import KRG
from smt.sampling_methods import LHS
import pyDOE2 as DOE

class asm_result:
    def __init__(self):
        self._result = []
        self._x_predicted = []
        self._f_predicted = []
        self._c_predicted = []
        self._cv_predicted = []
        self._x_true = []
        self._f_true = []
        self._c_true = []
        self._cv_true = []
    
    def get_result(self, iter):
        return self._result[iter-1]
    
    def set_result(self, result, iter=-1):
        '''
        iter = -1: append result in the self.result list
        iter = 0 or len(self.result): overwrite result of latest iteration
        iter < len(self.result): overwrite result of specified iteration
        '''
        len_result = len(self._result)
        constr = [v for val in result.constr for v in val]
        constr = np.array(constr[0:-result.x.shape[0]])
        if (iter == len_result + 1) or (iter == -1):
            self._result.append(result)
            self._x_predicted.append(result.x)
            self._f_predicted.append(result.fun)
            self._c_predicted.append(constr)
        elif (iter == len_result) or (iter == 0):
            self._result[-1] = result
        elif iter < len_result:
            self._result[iter-1] = result
        else:
            raise ValueError('iteration number wrong.')
            
    result = property(get_result, set_result)
    

class model(object):
    def __init__(self, **kwargs):
        self._x = []
        self._xlimits = []
        self._f = []
        self._sm = []
    '''
    self._x
    multi-level design points
    '''
    def get_x(self, level=-1):
        '''
        obtain x value at specified level.
        level>0: specific level, level=0: deepest level, level=-1: combine x in all levels
        '''
        len_x = len(self._x)
        if len_x == 0:
            # nothing to return
            return None
        if level == -1:
            # combine x in all levels and return
            m_var = self._x[0].shape[1]
            x = np.zeros((0,m_var), dtype=float)
            for idx in range(0, len(self._x)):
                x = np.append(x, self._x[idx], axis=0)
            return x
        elif level == 0:
            # return x in the deepest level
            return self._x[len_x-1]
        elif level > 0:
            # return x in the specified level
            if level <= len_x:
                return self._x[level-1]
            else:
                raise ValueError('level should not be greater than existing level (%d).' % len_x)
        else:
            raise ValueError('level should be either -1, 0, or greater than 0.\n' \
                + 'level>0: specific level, level=0: deepest level, level=-1: combine x in all levels.')
        return None
    def set_x(self, x, xlimits=None, level=0):
        '''
        x will be appended to object.x at specified level.
        x should be in two-dimensional array with a size of (n_x=n_f, n_var).
        xlimits defines lb and ub, in np.array([[LB1, UB1], [LB2, UB2], ...]) format.
        level>0: specified level, level=0: deepest level, level=-1: create next level.
        '''
        # determine length of levels
        if type(self._x) == list:
            len_x = len(self._x)
        elif type(self._x) == np.ndarray:
            self._x = [self._x]
            len_x = len(self._x)
        else:
            self._x = []
            len_x = len(self._x)
        if type(self._xlimits) == list:
            len_xlimits = len(self._xlimits)
        elif type(self._xlimits) == np.ndarray:
            self._xlimits = [self._xlimits]
            len_xlimits = len(self._xlimits)
        else:
            self._xlimits = []
            len_xlimits = len(self._xlimits)
        # test if levels of x and xlimits have the same length.
        if not (len_x == len_xlimits):
            raise ValueError('levels of x and xlimits are different.')
        # make type of x np.ndarray
        if (type(x) == np.ndarray) and (len(x.shape) == 2):
            x_new = x
        elif (type(x) == list) and (len(np.array(x, dtype=float).shape) == 2):
            x_new = np.array(x, dtype=float)
        else:
            raise ValueError('type of x should be numpy ndarray with dimension (n_x, n_var).')
        # make type of xlimits np.ndarray
        if (type(xlimits) == np.ndarray) and (len(xlimits.shape) == 2):
            xlimits_new = xlimits
        elif (type(xlimits) == list) and (len(np.array(xlimits, dtype=float).shape) == 2):
            xlimits_new = np.array(xlimits, dtype=float)
        else:
            raise ValueError('type of xlimits should be numpy ndarray with dimension (n_var, 2).')
        # add x into object.x
        if len_x == 0:
            # initialize
            self._x.append(x_new)
            self._xlimits.append(xlimits_new)
        elif level == -1:
            # create new level (one-level deeper than current deepest level) for new points
            if (((x_new.shape[1] == self._x[-1].shape[1])
            and (xlimits_new.shape[0] == self._xlimits[-1].shape[0])
            ) and (xlimits_new.shape[0] == x_new.shape[1])):
                self._x.append(x_new)
                self._xlimits.append(xlimits_new)
            else:
                raise ValueError('number of variables (n_var) should match with previous points.')
        elif level == 0:
            # add new points to the current deepest level
            if (((x_new.shape[1] == self._x[-1].shape[1])
            and (xlimits_new.shape[0] == self._xlimits[-1].shape[0])
            ) and (xlimits_new.shape[0] == x_new.shape[1])):
                n_var = xlimits_new[0]
                self._x[-1] = np.append(self._x[-1], x_new, axis=0)
                self._xlimits[-1] = np.append(
                    np.min(
                        np.append(
                            self._xlimits[-1][:,0].reshape(n_var, 1),
                            xlimits_new[:,0].reshape(n_var, 1),
                            axis = 1
                        ),
                        axis = 1
                    ).reshape(n_var, 1),
                    np.max(
                        np.append(
                            self._xlimits[-1][:,1].reshape(n_var, 1),
                            xlimits_new[:,1].reshape(n_var, 1),
                            axis = 1
                        ),
                        axis = 1
                    ).reshape(n_var, 1),
                    axis = 1
                )
            else:
                raise ValueError('number of variables (n_var) should match with previous points.')
        elif level > 0:
            # add new points to the specified level
            if (((x_new.shape[1] == self._x[-1].shape[1])
            and (xlimits_new.shape[0] == self._xlimits[-1].shape[0])
            ) and (xlimits_new.shape[0] == x_new.shape[1])):
                n_var = xlimits_new[0]
                self._x[level-1] = np.append(self._x[level-1], x_new, axis=0)
                self._xlimits[level-1] = np.append(
                    np.min(
                        np.append(
                            self._xlimits[level-1][:,0].reshape(n_var, 1),
                            xlimits_new[:,0].reshape(n_var, 1),
                            axis = 1
                        ),
                        axis = 1
                    ).reshape(n_var, 1),
                    np.max(
                        np.append(
                            self._xlimits[level-1][:,1].reshape(n_var, 1),
                            xlimits_new[:,1].reshape(n_var, 1),
                            axis = 1
                        ),
                        axis = 1
                    ).reshape(n_var, 1),
                    axis = 1
                )
            else:
                raise ValueError('number of variables (n_var) should match with previous points.')
        else:
            raise ValueError('level should be either -1, 0, or greater than 0.\n' \
                + 'level>0: specific level, level=0: deepest level, level=-1: create next level.')
        return
    def del_x(self, level=0, i_point=-1):
        '''
        x of specified level will be deleted.
        level>0: specific level, level=0: deepest level, level=-1: all levels.
        i_point>0: specific point, i_point=0: last point, i_point=-1: all points.
        '''
        # remove empty level
        for idx in range(0, len(self._x)):
            jdx = len(self._x) - idx - 1
            if self._x[jdx].shape[0] == 0:
                self._x.pop(jdx)
        len_x = len(self._x)
        if len_x == 0:
            # if nothing is in x, re-initialize
            self._x = []
            return
        if level == -1:
            # points from all levels
            if i_point == -1:
                # remove all points from all levels
                self._x = []
                return
            elif i_point == 0:
                # remove last point from all levels (equivalent to last point from deepest level)
                if self._x[-1].shape[0] == 1:
                    # if only one row exists
                    self._x.pop()
                    return
                else:
                    # delete last point
                    self._x[-1] = np.delete(self._x[-1], -1, axis=0)
                    return
            elif i_point > 0:
                # remove specified point from all levels
                n_points = 0
                for idx in range(0, len(self._x)):
                    n_points += self._x[idx].shape[0]
                if i_point <= n_points:
                    j_point = 0
                    for idx in range(0, len(self._x)):
                        j_point += self._x[idx].shape[0]
                        if i_point <= j_point:
                            k_point = self._x[idx].shape[0] - (j_point - i_point) - 1
                            self._x[idx] = np.delete(self._x[idx], k_point, axis=0)
                            return
                else:
                    raise ValueError('i_point should be less than or equal to the total number of points.')
        elif level == 0:
            # points from deepest level
            if i_point == -1:
                # remove all points from the deepest level
                self._x.pop()
                self._xlimits.pop()
                return
            elif i_point == 0:
                # remove last point from the deepest level
                if self._x[-1].shape[0] == 1:
                    # if only one row exists
                    self._x.pop()
                    self._xlimits.pop()
                    return
                else:
                    # delete last point
                    self._x[-1] = np.delete(self._x[-1], -1, axis=0)
                    return
            elif i_point > 0:
                # remove specified point from the deepest level
                if i_point <= self._x[-1].shape[0]:
                    if (self._x[-1].shape[0] == 1) and (i_point == 1):
                        # if only one row exists and deleting the only point
                        self._x.pop()
                        self._xlimits.pop()
                    else:
                        self._x[-1] = np.delete(self._x[-1], i_point-1, axis=0)
                    return
                else:
                    raise ValueError('i_point should be less than or equal to the number of points in the deepest level.')
            else:
                raise ValueError('i_point should be either -1, 0, or positive number.\n' \
                    + 'i_point>0: specific point, i_point=0: last point, i_point=-1: all points.')
        elif level > 0:
            # points from specified level
            if level > len(self._x):
                raise('level should be less than or equal to number of existing levels.')
            if i_point == -1:
                # remove all points from specified level
                self._x.pop(level-1)
                self._xlimits.pop(level-1)
                return
            if i_point == 0:
                # remove last point from specified level
                if self._x[level-1].shape[0] == 1:
                    self._x.pop(level-1)
                    self._xlimits.pop(level-1)
                    return
                else:
                    self._x[level-1] = np.delete(self._x[level-1], -1, axis=0)
                    return
            if i_point > 0:
                # remove specified point from specified level
                if i_point <= self._x[level-1].shape[0]:
                    if (self._x[level-1].shape[0] == 1) and (i_point == 1):
                        self._x.pop(level-1)
                        self._xlimits.pop(level-1)
                        return
                    else:
                        self._x[level-1] = np.delete(self._x[level-1], i_point-1, axis=0)
                        return
                else:
                    raise ValueError('i_point should be less than or equal to the number of points in the specified level.')
            else:
                raise('i_point should be either -1, 0, or positive number.\n' \
                    + 'i_point>0: specific point, i_point=0: last point, i_point=-1: all points.')
        else:
            raise ValueError('level should be either -1, 0, or greater than 0.\n' \
                + 'level>0: specific level, level=0: deepest level, level=-1: all levels.')
    # register getter, setter, and deleter for object.x
    x = property(get_x, set_x, del_x)
    '''
    self._f
    function values of multi-level design points
    '''
    def get_f(self, level=-1):
        '''
        obtain f value at specified level.
        level>0: specific level, level=0: deepest level, level=-1: combine f in all levels
        '''
        len_f = len(self._f)
        if len_f == 0:
            # nothing to return
            return None
        if level == -1:
            # combine f in all levels and return
            m_fun = self._f[0].shape[1]
            f = np.zeros((0, m_fun), dtype=float)
            for idx in range(0, len(self._f)):
                f = np.append(f, self._f[idx], axis=0)
            return f
        elif level == 0:
            # return f in the deepest level
            return self._f[len_f-1]
        elif level > 0:
            # return f in the specified level
            if level <= len_f:
                return self._f[level-1]
            else:
                raise ValueError('level should not be greater than existing level (%d).' % len_f)
        else:
            raise ValueError('level should be either -1, 0, or greater than 0.\n' \
                + 'level>0: specific level, level=0: deepest level, level=-1: combine f in all levels.')
        return
    def set_f(self, f, level=0):
        '''
        f will be appended to object.f at specified level.
        f should be in two-dimensional array with a size of (n_x=n_f, n_fun).
        level>0: specified level, level=0: deepest level, level=-1: create next level
        '''
        # determine length of levels
        if type(self._f) == list:
            len_f = len(self._f)
        elif type(self._f) == np.ndarray:
            self._f = [self._f]
            len_f = len(self._f)
        else:
            self._f = []
            len_f = len(self._f)
        # make type of f nd.ndarray
        if (type(f) == np.ndarray) and (len(f.shape) == 2):
            f_new = f
        elif (type(f) == list) and (len(np.array(f, dtype=float).shape) == 2):
            f_new = np.array(f, dtype=float)
        else:
            raise ValueError('type of f should be numpy ndarray with dimension (n_f, n_fun).')
        # add f into object.f
        if len_f == 0:
            # initialize
            self._f.append(f_new)
        elif level == -1:
            # create new level (one-level deeper than current deepest level) for new points
            if f_new.shape[1] == self._f[-1].shape[1]:
                self._f.append(f_new)
            else:
                raise ValueError('number of functions (n_fun) should match with previous points.')
        elif level == 0:
            # add new points to the current deepest level
            if f_new.shape[1] == self._f[-1].shape[1]:
                self._f[-1] = np.append(self._f[-1], f_new, axis=0)
            else:
                raise ValueError('number of functions (n_fun) should match with previous points.')
        elif level > 0:
            # add new points to the specified level
            if f_new.shape[1] == self._f[-1].shape[1]:
                self._f[level-1] = np.append(self._f[level-1], f_new, axis=0)
            else:
                raise ValueError('number of functions (n_fun) should match with previous points.')
        else:
            raise ValueError('level should be either -1, 0, or greater than 0.\n' \
                + 'level>0: specific level, level=0: deepest level, level=-1: create next level.')
        return
    def del_f(self, level=0, i_point=-1):
        '''
        f of specified level will be deleted.
        level>0: specific level, level=0: deepest level, level=-1: all levels.
        i_point>0: specific point, i_point=0: last point, i_point=-1: all points.
        '''
        # remove empty level
        for idx in range(0, len(self._f)):
            jdx = len(self._f) - idx - 1
            if self._f[jdx].shape[0] == 0:
                self._f.pop(jdx)
        len_f = len(self._f)
        if len_f == 0:
            # if nothing is in f, re-initialize
            self._f = []
            return
        if level == -1:
            # points from all levels
            if i_point == -1:
                # remove all points from all levels
                self._f = []
                return
            elif i_point == 0:
                # remove last point from all levels (equivalent to last point from deepest level)
                if self._f[-1].shape[0] == 1:
                    # if only one row exists
                    self._f.pop()
                    return
                else:
                    # delete last point
                    self._f[-1] = np.delete(self._f[-1], -1, axis=0)
                    return
            elif i_point > 0:
                # remove specified point from all levels
                n_points = 0
                for idx in range(0, len(self._f)):
                    n_points += self._f[idx].shape[0]
                if i_point <= n_points:
                    j_point = 0
                    for idx in range(0, len(self._f)):
                        j_point += self._f[idx].shape[0]
                        if i_point <= j_point:
                            k_point = self._f[idx].shape[0] - (j_point - i_point) - 1
                            self._f[idx] = np.delete(self._f[idx], k_point, axis=0)
                            return
                else:
                    raise ValueError('i_point should be less than or equal to the total number of points.')
        elif level == 0:
            # points from deepest level
            if i_point == -1:
                # remove all points from the deepest level
                self._f.pop()
                return
            elif i_point == 0:
                # remove last point from the deepest level
                if self._f[-1].shape[0] == 1:
                    # if only one row exists
                    self._f.pop()
                    return
                else:
                    # delete last point
                    self._f[-1] = np.delete(self._f[-1], -1, axis=0)
                    return
            elif i_point > 0:
                # remove specified point from the deepest level
                if i_point <= self._f[-1].shape[0]:
                    self._f[-1] = np.delete(self._f[-1], i_point-1, axis=0)
                    return
                else:
                    raise ValueError('i_point should be less than or equal to the number of points in the deepest level.')
            else:
                raise ValueError('i_point should be either -1, 0, or positive number.\n' \
                    + 'i_point>0: specific point, i_point=0: last point, i_point=-1: all points.')
        elif level > 0:
            # points from specified level
            if level > len(self._f):
                raise('level should be less than or equal to number of existing levels.')
            if i_point == -1:
                # remove all points from specified level
                self._f.pop(level-1)
                return
            if i_point == 0:
                # remove last point from specified level
                if self._f[level-1].shape[0] == 1:
                    self._f.pop(level-1)
                    return
                else:
                    self._f[level-1] = np.delete(self._f[level-1], -1, axis=0)
                    return
            if i_point > 0:
                # remove specified point from specified level
                if i_point > self._f[level-1].shape[0]:
                    raise ValueError('i_point should be less than or equal to the number of points in the specified level.')
                else:
                    if self._f[level-1].shape[0] == 1:
                        self._f.pop(level-1)
                        return
                    else:
                        self._f[level-1] = np.delete(self._f[level-1], i_point-1, axis=0)
                        return
            else:
                raise('i_point should be either -1, 0, or positive number.\n' \
                    + 'i_point>0: specific point, i_point=0: last point, i_point=-1: all points.')
        else:
            raise ValueError('level should be either -1, 0, or greater than 0.\n' \
                + 'level>0: specific level, level=0: deepest level, level=-1: all levels.')
    # register getter, setter, and deleter for object.f
    f = property(get_f, set_f, del_f)
    '''
    self._sm
    list of surrogate model associated with object.x and object.f
    '''
    def train_sm(self, level=0, option=None):
        '''
        train surrogate model.
        level=0: deepest level, level=-1: combine f in all levels
        '''
        len_x = len(self._x)
        len_f = len(self._f)
        len_sm = len(self._sm)
        if not (len_x == len_f):
            raise ValueError('number of levels in x and f are different.')
        if len_x == 0:
            raise ValueError('no points exists for training.')
        if level == -1:
            # train model for points from all levels
            sm = []
            for idx in range(0, len_x):
                sm_tmp = KRG(print_global=False)
                x = deepcopy(self._x[idx])
                f = deepcopy(self._f[idx])
                xlimits = deepcopy(self._xlimits[idx])
                if idx > 0:
                    for jdx in reversed(range(0,idx)):
                        xn = deepcopy(self._x[jdx])
                        fn = deepcopy(self._f[jdx])
                        for kdx in range(0, xlimits.shape[0]):
                            m1 = xn[:,kdx]>=xlimits[kdx,0]
                            m2 = xn[:,kdx]<=xlimits[kdx,1]
                            m0 = np.logical_and(m1, m2)
                            xn = xn[m0, :]
                            fn = fn[m0, :]
                        x = np.append(x, xn, axis=0)
                        f = np.append(f, fn, axis=0)
                for jdx in range(0, idx):
                    f -= sm[jdx].predict_values(x)
                sm_tmp.set_training_values(x, f)
                sm_tmp.train()
                sm.append(deepcopy(sm_tmp))
                del(sm_tmp)
            self._sm = sm
        elif level == 0:
            # points from deepest level
            sm = self._sm
            if len_sm == len_x:
                sm[-1] = None
            elif len_sm + 1 == len_x:
                sm.append(None)
            else:
                raise ValueError('sm dimension wrong.')
            idx_KRG = 0
            for idx in range(0, len_sm):
                if type(sm[idx]) == KRG:
                    idx_KRG = idx + 1
                else:
                    break
            for idx in range(idx_KRG, len_x):
                sm_tmp = KRG(print_global=False)
                x = deepcopy(self._x[idx])
                f = deepcopy(self._f[idx])
                xlimits = deepcopy(self._xlimits[idx])
                if idx > 0:
                    for jdx in reversed(range(0,idx)):
                        xn = deepcopy(self._x[jdx])
                        fn = deepcopy(self._f[jdx])
                        for kdx in range(0, xlimits.shape[0]):
                            m1 = xn[:,kdx]>=xlimits[kdx,0]
                            m2 = xn[:,kdx]<=xlimits[kdx,1]
                            m0 = np.logical_and(m1, m2)
                            xn = xn[m0, :]
                            fn = fn[m0, :]
                        x = np.append(x, xn, axis=0)
                        f = np.append(f, fn, axis=0)
                for jdx in range(0, idx):
                    f -= sm[jdx].predict_values(x)
                sm_tmp.set_training_values(x, f)
                sm_tmp.train()
                sm[idx] = deepcopy(sm_tmp)
                del(sm_tmp)
            self._sm = sm
        else:
            raise ValueError('level should be either -1 or 0.\n' \
                + 'level=0: deepest level, level=-1: all levels.')
    def predict_sm(self, x, level=0):
        '''
        predict function using surrogate model at specified level.
        level>0: using models up to specified level,
        level=0: using models up to deepest possible level.
        '''
        len_sm = len(self._sm)
        if len_sm == 0:
            raise ValueError('surrogate model is empty.')
        if type(x) == list:
            # list to np.ndarray
            x = np.array(x)
            if (len(x.shape) == 1) and (self._sm[0].nx == 1):
                x = x.reshape(x.shape[0],1)
        if not (x.shape[1] == self._sm[0].nx):
            raise ValueError('n_var dimension of x is incorrect.')
        # determine levels to explore
        if level == 0:
            # predict values using models up to deepest possible level
            level_max = len_sm
        elif level > 0:
            # predict values using models up to specified level
            level_max = np.minimum(level, len_sm)
        else:
            raise ValueError('level should be 0 or positive.\n' \
                + 'level=0: up to deepest level, level>0: up to specified level.')
        # predict values
        f = np.zeros((x.shape[0], self._sm[0].ny), dtype=float)
        for idx in range(0, level_max):
            fn = self._sm[idx].predict_values(x)
            for kdx in range(0, self._xlimits[idx].shape[0]):
                m1 = x[:,kdx]>=self._xlimits[idx][kdx,0]
                m2 = x[:,kdx]<=self._xlimits[idx][kdx,1]
                m0 = np.logical_not(np.logical_and(m1, m2))
                fn[m0, :] = 0.0
            f += fn
        return f
    '''
    sampling method
    '''
    def sampling(self, nx, xlimits, method='LHS'):
        '''
        create nx samples bounded by xlimits using specified method.
        xlimits defines lb and ub, in np.array([[LB1, UB1], [LB2, UB2], ...]) format.
        method = 'LHS': Latin hypercube sampling, 'CCD': centralized composite design,
                 'PBD': Plackett-Burman design, 'PB-CCD': Plackett-Burman centralized composite design
        '''
        n_var = xlimits.shape[0]
        # Sampling
        if method.lower() == 'lhs':
            x = DOE.lhs(n_var, samples=nx, criterion='correlation')*2.0 - 1.0
        elif method.lower() == 'ccd':
            if n_var > 8:
                raise ValueError('number of variables is TOO LARGE for centralized composite design (CCD).')
            if n_var > 7:
                warnings.warn('number of variables is TOO LARGE for centralized composite design (CCD).')
            x = DOE.ccdesign(n_var, center=(0,1), alpha='rotatable', face='inscribed')
        elif method.lower() == 'pbd':
            x = DOE.pbdesign(n_var)
        elif method.lower() in ['pb-ccd', 'pbccd']:
            l = np.sqrt(n_var)
            x = DOE.pbdesign(n_var)/l
            x = np.append(x, -x/2.0, axis=0)
            for idx in range(0, n_var):
                z = np.zeros((1,n_var))
                z[0,idx] = 1.0
                x = np.append(x, z, axis=0)
                z[0,idx] = -1.0
                x = np.append(x, z, axis=0)
            x = np.append(x, np.zeros((1, n_var)), axis=0)
        # Scale
        for idx in range(0, xlimits.shape[0]):
            x[:,idx] = (x[:,idx] + 1.0)/2.0*(xlimits[idx,1] - xlimits[idx,0]) + xlimits[idx,0]
        # Return
        return x





def six_hump_camel_function(x):
    f = (4.0 - 2.1*x[:,0]**2 + x[:,0]**4/3.0)*x[:,0]**2 + x[:,0]*x[:,1] + (-4.0 + 4.0*x[:,1]**2)*x[:,1]**2
    return f.reshape(x.shape[0],1)

if __name__ == '__main__':
    a = model()
    xlimits1 = np.array([[-5.0, 5.0],[-2.0,2.0]])
    #smp1 = LHS(xlimits=xlimits1, criterion='centermaximin')
    #x1 = smp1(50)
    x1 = a.sampling(None, xlimits1, method='PB-CCD')
    f1 = six_hump_camel_function(x1)
    a.set_x(x1, xlimits=xlimits1, level=-1)
    a.set_f(f1, level=-1)
    xlimits2 = np.array([[-2.0, 2.0],[-1.0,1.0]])
    #smp2 = LHS(xlimits=xlimits2, criterion='centermaximin')
    #x2 = smp2(50)
    x2 = a.sampling(30, xlimits2, method='LHS')
    f2 = six_hump_camel_function(x2)
    a.set_x(x2, xlimits=xlimits2, level=-1)
    a.set_f(f2, level=-1)
    a.train_sm(level=-1)
    from matplotlib import pyplot as plt
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.scatter(x1[:,0],x1[:,1],f1[:,0],marker='o')
    X1 = np.linspace(xlimits1[0,0], xlimits1[0,1], 21)
    Y1 = np.linspace(xlimits1[1,0], xlimits1[1,1], 21)
    X1, Y1 = np.meshgrid(X1, Y1)
    V1 = np.append(
        X1.reshape((X1.size, 1)),
        Y1.reshape((Y1.size, 1)),
        axis = 1
    )
    Z1 = a.predict_sm(V1)
    Z1 = Z1.reshape((X1.shape[0],X1.shape[1]))
    ax1.plot_surface(X1, Y1, Z1, alpha=0.1)
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.scatter(x2[:,0],x2[:,1],f2[:,0],marker='x')
    X2 = np.linspace(xlimits2[0,0], xlimits2[0,1], 21)
    Y2 = np.linspace(xlimits2[1,0], xlimits2[1,1], 21)
    X2, Y2 = np.meshgrid(X2, Y2)
    V2 = np.append(
        X2.reshape((X2.size, 1)),
        Y2.reshape((Y2.size, 1)),
        axis = 1
    )
    Z2 = a.predict_sm(V2)
    Z2 = Z2.reshape((X2.shape[0],X2.shape[1]))
    ax2.plot_surface(X2, Y2, Z2, alpha=0.1)
    plt.show()
