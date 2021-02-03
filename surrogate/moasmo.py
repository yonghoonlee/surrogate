#!/usr/bin/env python
# coding: utf-8
# Copyright (c) 2021 Yong Hoon Lee

import numpy as np

class model(object):
    def __init__(self, **kwargs):
        self._x = []
        self._xlb = None
        self._xub = None
        self._f = []
        self._flb = None
        self._fub = None
        self._sm = None
        self.scale_x = True
        self.scale_f = True
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
        return
    def set_x(self, x, level=0):
        '''
        x will be appended to object.x at specified level.
        x should be in two-dimensional array with a size of (n_x=n_f, n_var).
        level>0: specified level, level=0: deepest level, level=-1: create next level
        '''
        len_x = len(self._x)
        # make type of x np.ndarray
        if (type(x) == np.ndarray) and (len(x.shape) == 2):
            x_new = x
        elif (type(x) == list) and (len(np.array(x, dtype=float).shape) == 2):
            x_new = np.array(x, dtype=float)
        else:
            raise ValueError('type of x should be numpy ndarray with dimension (n_x, n_var).')
        # add x into object.x
        if len_x == 0:
            # initialize
            self._x.append(x_new)
        elif level == -1:
            # create new level (one-level deeper than current deepest level) for new points
            if x_new.shape[1] == self._x[-1].shape[1]:
                self._x.append(x_new)
            else:
                raise ValueError('number of variables (n_var) should match with previous points.')
        elif level == 0:
            # add new points to the current deepest level
            if x_new.shape[1] == self._x[-1].shape[1]:
                self._x[-1] = np.append(self._x[-1], x_new, axis=0)
            else:
                raise ValueError('number of variables (n_var) should match with previous points.')
        elif level > 0:
            # add new points to the specified level
            if x_new.shape[1] == self._x[-1].shape[1]:
                self._x[level-1] = np.append(self._x[level-1], x_new, axis=0)
            else:
                raise ValueError('number of variables (n_var) should match with previous points.')
        else:
            raise ValueError('level should be either -1, 0, or greater than 0.\n' \
                + 'level>0: specific level, level=0: deepest level, level=-1: create next level.')
        print('level=%d' % level)
        print('x_new=')
        print(x_new)
        print('x=')
        print(self._x)
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
            if i_point == -1:
                # remove all points from the deepest level
                self._x.pop()
                return
            elif i_point == 0:
                # remove last point from the deepest level
                if self._x[-1].shape[0] == 1:
                    # if only one row exists
                    self._x.pop()
                    return
                else:
                    # delete last point
                    self._x[-1] = np.delete(self._x[-1], -1, axis=0)
                    return
            elif i_point > 0:
                # remove specified point from the deepest level
                if i_point <= self._x[-1].shape[0]:
                    self._x[-1] = np.delete(self._x[-1], i_point-1, axis=0)
                    return
                else:
                    raise ValueError('i_point should be less than or equal to the number of points in the deepest level.')
            else:
                raise ValueError('i_point should be either -1, 0, or positive number.\n' \
                    + 'i_point>0: specific point, i_point=0: last point, i_point=-1: all points.')
        elif level > 0:
            if level > len(self._x):
                raise('level should be less than or equal to number of existing levels.')
            if i_point == -1:
                # remove all points from specified level
                self._x.pop(level-1)
                return
            if i_point == 0:
                # remove last point from specified level
                if self._x[level-1].shape[0] == 1:
                    self._x.pop(level-1)
                    return
                else:
                    self._x[level-1] = np.delete(self._x[level-1], -1, axis=0)
                    return
            if i_point > 0:
                # remove specified point from specified level
                if i_point > self._x[level-1].shape[0]:
                    raise ValueError('i_point should be less than or equal to the number of points in the specified level.')
                else:
                    if self._x[level-1].shape[0] == 1:
                        self._x.pop(level-1)
                        return
                    else:
                        self._x[level-1] = np.delete(self._x[level-1], i_point-1, axis=0)
                        return
            else:
                raise('i_point should be either -1, 0, or positive number.\n' \
                    + 'i_point>0: specific point, i_point=0: last point, i_point=-1: all points.')
        else:
            raise ValueError('level should be either -1, 0, or greater than 0.\n' \
                + 'level>0: specific level, level=0: deepest level, level=-1: all levels.')
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
        



