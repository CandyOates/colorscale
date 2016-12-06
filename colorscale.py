from scipy import stats
import functools


class RGB(object):
    def __init__(self, rgb):
        '''
        :param rgb: tuple of length 3
        '''
        assert len(rgb) == 3 and all([isinstance(x, int) for x in rgb])
        assert all([x >= 0 and x <= 255 for x in rgb])
        self._rgb = rgb

    @property
    def rgb(self):
        return self._rgb

    @staticmethod
    def from_hex(hx):
        assert hx[0] == '#' and len(hx) == 7
        r = int(hx[1:3], 16)
        b = int(hx[3:5], 16)
        g = int(hx[5:], 16)
        return RGB((r, g, b))

    @staticmethod
    def from_name(name):
        if name.lower() == 'red':
            return RGB((255, 0, 0))
        elif name.lower() == 'green':
            return RGB((0, 255, 0))
        elif name.lower() == 'blue':
            return RGB((0, 0, 255))
        elif name.lower() == 'black':
            return RGB((0, 0, 0))
        elif name.lower() == 'white':
            return RGB((255, 255, 255))
        raise RuntimeError('Invalid color name {0}'.format(name))

    def __repr__(self):
        return 'rgb({0}, {1}, {2})'.format(*self.rgb)

    def __str__(self):
        return repr(self)

    @property
    def hex(self):
        r, g, b = [hex(x)[2:] for x in self.rgb]
        hx = '#{0}{1}{2}'.format(
            r if len(r) == 2 else '0{0}'.format(r),
            g if len(g) == 2 else '0{0}'.format(g),
            b if len(b) == 2 else '0{0}'.format(b))
        return hx

    @property
    def dec(self):
        return tuple([x/255. for x in self.rgb])


def __mult(v, x):
    return [x*y for y in v]


def __add(x, y):
    assert len(x) == len(y)
    return [x[k] + y[k] for k in range(len(x))]


def colorscale(start, end, mid=None, npoints=10):
    if mid:
        n = npoints / 2
        b = (npoints + 1) % 2
        scale1 = colorscale(start, mid, npoints=n)
        scale2 = colorscale(mid, end, npoints=n+b)
        return scale1 + scale2[1:]
    else:
        inps = dict(start=start, end=end)
        for k, v in inps.iteritems():
            if isinstance(v, list) or isinstance(v, tuple):
                inps[k] = RGB(v)
            elif isinstance(v, str) or isinstance(v, unicode):
                if v[0] == '#':
                    inps[k] = RGB.from_hex(v)
                else:
                    inps[k] = RGB.from_name(v)
            elif isinstance(v, RGB):
                pass
            else:
                raise RuntimeError(
                    'Invalid type for {0} in colorscale'.format(k))
        p0 = inps['start'].rgb
        p1 = inps['end'].rgb
        dt = 1. / (npoints - 1)
        dp = [y-x for x, y in zip(p0, p1)]
        scale = [__add(p0, __mult(dp, k*dt)) for k in range(npoints)]
        scale = [RGB([int(x) for x in y]) for y in scale]
        return scale


def linear_kernel(n, a, b):
    dt = (b - a) / (n - 1.)
    rng = [a + k*dt for k in range(n)]
    return rng


def gaussian_kernel(n, mu, sig, nstds=3):
    a = mu - nstds*sig
    b = mu + nstds*sig

    cdf = functools.partial(stats.norm.cdf, loc=mu, scale=sig)
    icdf = functools.partial(stats.norm.ppf, loc=mu, scale=sig)
    pr = cdf(b) - cdf(a)
    dt = pr / (n - 1.)

    rng = [a]
    for k in range(n-1):
        v = icdf(cdf(rng[-1]) + dt)
        rng.append(v)
    return rng


class Colorscale(object):
    def __init__(self, start, end, mid=None, npoints=10,
                 kernel=linear_kernel, kernelargs=None,
                 kernelkwargs=None):
        '''
        :param start: RGB or hex or color name
        :param end: RGB or hex or color name
        :param mid: (optional) RGB or hex or color name
        :param npoints: int, granularity of gradient
        :param kernel: function, colors kernel
        :param kernelargs: args for the kernel (minus n)
        :param kernelkwargs: kwargs for the kernel (minus n)
        '''
        args = kernelargs or tuple()
        kwargs = kernelkwargs or dict()
        self._scale = colorscale(start, end, mid, npoints)
        self._rng = kernel(npoints, *args, **kwargs)

    @property
    def scale(self):
        return self._scale

    @property
    def range(self):
        return self._rng

    def rgb(self, val):
        '''
        :param val: number

        Get RGB from colorscale for val.
        '''
        inds = [k for k, x in enumerate(self.range) if val <= x]
        ind = inds[0] if inds else -1
        clr = self.scale[ind]
        return clr
