#!/usr/bin/env python
# coding: utf-8
# Copyright (c) 2021 Yong Hoon Lee

class model(object):
    def __init__(self, **kwargs):
        self._x = np.zeros((1,0,0))
        self._xlb = None
        self._xub = None
        self._f = None
        self._flb = None
        self._fub = None
        self._sm = None
        self.scale_x = True
        self.scale_f = True
    def get_x(self, level=0):
        '''
        obtain x value at specified level.
        level>0: specific level, level=0: deepest level, level=-1: combine x in all levels
        '''
        if len(self._x.shape) == 3:
            if level == -1: # all levels
                n_level = self._x.shape[0]
                if n_level in [0, 1]:
                    return self._x
                x = self._x[0]
                for idx in range(1, n_level):
                    x = np.append(x, self._x[idx], axis=0)
                return x
            elif level == 0: # deepest level
                n_level = self._x.shape[0]
                if n_level in [0, 1]:
                    return self._x
                else:
                    return np.array([self._x[n_level-1]])
            elif level > 0:
                n_level = self._x.shape[0]
                if n_level >= level:
                    return np.array([self._x[level-1]])
                else:
                    raise ValueError('level %d does not exist.' % level)
            else:
                raise ValueError('level should be 0 (all) or positive.')
        else:
            raise Exception('dimension of x should be (n_level, n_x, n_var).')
    def set_x(self, x_new, level=0):
        '''
        x_new will be appended to x at specified level.
        level>0: specific level, level=0: deepest level, level=-1: create next level
        '''
        if level == 0:
            i_level = self._x.shape[0] - 1
            self._x[i_level] = np.append(self._x[i_level], x_new, axis=0)
        elif level == -1:
            pass
        elif level > 0:
            pass
        else:
            raise ValueError('level should be either -1, 0, or greater than 0.')