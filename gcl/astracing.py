import time
import logging

_enabled = False
_with_logging = False
traces = None
levels = []


logger = logging.getLogger(__name__)


class Invocation(object):
  def __init__(self, time, count=1):
    self.count = count
    self.time = time

  def __iadd__(self, rhs):
    self.count += rhs.count
    self.time += rhs.time
    return self

  def __str__(self):
    return '%8.3f/%3d (%.4f)' % (self.time, self.count, self.time/self.count)


class ExpressionTrace(object):
  def __init__(self):
    self.succeeded_parsing = Invocation(0.0)
    self.failed_parsing = Invocation(0.0)

  @property
  def total_time(self):
    return self.succeeded_parsing.time + self.failed_parsing.time


def enable(do_trace=True, do_logging=False):
  global _enabled, traces, _with_logging
  _enabled = do_trace
  _with_logging = do_logging
  import collections
  traces = collections.defaultdict(ExpressionTrace)


def debugStart( instring, loc, expr ):
  levels.append((str(expr), time.time()))
  prefix = '  ' * len(levels)
  if _with_logging:
    logger.debug('%s{{{ %r trying %s', prefix, snippet(instring, loc), expr)


def debugSuccess( instring, startloc, endloc, expr, toks ):
  prefix = '  ' * len(levels)

  name, start_time = levels.pop()
  assert name == str(expr)
  delta = time.time() - start_time
  traces[str(expr)].succeeded_parsing += Invocation(delta)
  if _with_logging:
    logger.debug('%s}}} %r success matching %s => %r, time=%s', prefix, snippet(instring, startloc), expr, toks, delta)


def debugException( instring, loc, expr, exc ):
  prefix = '  ' * len(levels)

  name, start_time = levels.pop()
  assert name == str(expr)
  delta = time.time() - start_time
  traces[str(expr)].failed_parsing += Invocation(delta)
  if _with_logging:
    logger.debug('%s}}} %r failed %s, time=%s', prefix, snippet(instring, loc), expr, delta)


def maybe_trace(expr):
  if _enabled:
    expr.setDebugActions(debugStart, debugSuccess, debugException)


def print_traces():
  lines = list(traces.items())
  lines.sort(key=lambda t: t[1].total_time, reverse=True)

  fmt = '%-20s  %23s  %23s'

  print fmt % ('Name', 'Success', 'Failed')
  print fmt % ('---------', '-------', '------')

  for name, trace in lines:
    print fmt % (name, trace.succeeded_parsing, trace.failed_parsing)


def snippet(s, loc):
    return s[loc:loc+5] + '...'

