package edu.umass.cs.iesl.spdb

import cc.factorie.optimize._
import cc.factorie._
import cc.factorie.maths._
import collection.mutable.ArrayBuffer

/**
 * @author Sebastian Riedel
 */
class MyLimitedMemoryBFGS(val optimizable: OptimizableByValueAndGradient) extends Optimizer with FastLogging {

  case class StepTooSmallException(msg: String) extends Exception(msg)

  var isConverged = false
  var maxIterations = 1000
  var tolerance = 0.0001
  var gradientTolerance = 0.001
  var lineMaximizer = new BackTrackLineOptimizer(optimizable)

  val eps = 1.0e-5
  // The number of corrections used in BFGS update
  // ideally 3 <= m <= 7. Larger m means more cpu time, memory.
  val m = 4

  // State of search
  // g = gradient
  // s = list of m previous "parameters" values
  // y = list of m previous "g" values
  // rho = intermediate calculation
  var g: Array[Double] = null;
  var oldg: Array[Double] = null;
  var direction: Array[Double] = null;
  var params: Array[Double] = null;
  var oldParams: Array[Double] = null
  var s: ArrayBuffer[Array[Double]] = null;
  var y: ArrayBuffer[Array[Double]] = null
  var rho: ArrayBuffer[Double] = null
  var alpha: Array[Double] = null
  var step = 1.0
  var iterations: Int = 0

  // override to evaluate on dev set, save the intermediate model, etc.
  def postIteration(iter: Int): Unit = ()

  def optimize(numIterations: Int = Int.MaxValue): Boolean = {
    if (isConverged) return true;
    val initialValue = optimizable.optimizableValue
    val numParams = optimizable.numOptimizableParameters
    logger.info("LimitedMemoryBFGS: Initial value = " + initialValue);

    if (g == null) {
      // first time through
      iterations = 0
      s = new ArrayBuffer[Array[Double]]
      y = new ArrayBuffer[Array[Double]]
      rho = new ArrayBuffer[Double]
      alpha = new Array[Double](m)

      params = new Array[Double](numParams)
      oldParams = new Array[Double](numParams)
      g = new Array[Double](numParams)
      oldg = new Array[Double](numParams)
      direction = new Array[Double](numParams)

      // get the parameters
      optimizable.getOptimizableParameters(params)
      maths.set(oldParams, params)

      // get the gradient
      optimizable.getOptimizableGradient(g)
      maths.set(oldg, g)
      maths.set(direction, g)

      if (direction.absNormalize == 0) {
        logger.info("L-BFGS initial gradient is zero; saying converged");
        g = null
        isConverged = true
        return true;
      }
      direction *= 1.0 / direction.twoNorm

      // take a step in the direction
      step = lineMaximizer.optimize(direction, step)
      if (step == 0.0) {
        // could not step in this direction
        // give up and say converged
        g = null // reset search
        step = 1.0
        logger.error("Line search could not step in the current direction. " +
          "(This is not necessarily cause for alarm. Sometimes this happens close to the maximum," +
          " where the function may be very flat.)")
        //throw new StepTooSmallException("Line search could not step in current direction.")
        return false
      }

      optimizable.getOptimizableParameters(params)
      optimizable.getOptimizableGradient(g)
    }

    def pushArray(l: ArrayBuffer[Array[Double]], toadd: Array[Double]): Unit = {
      assert(l.size <= m)
      if (l.size == m) {
        val last = l(0)
        Array.copy(toadd, 0, last, 0, toadd.length)
        forIndex(l.size - 1)(i => {
          l(i) = l(i + 1)
        })
        l(m - 1) = last
      } else {
        val last = new Array[Double](toadd.length)
        Array.copy(toadd, 0, last, 0, toadd.length)
        l += last
      }
    }

    def pushDbl(l: ArrayBuffer[Double], toadd: Double): Unit = {
      assert(l.size <= m)
      if (l.size == m) l.remove(0)
      l += toadd
    }

    // step through iterations
    forIndex(numIterations)(iterationCount => {
      val value = optimizable.optimizableValue
      logger.info("LimitedMemoryBFGS: At iteration " + iterations + ", value = " + value);
      // get difference between previous 2 gradients and parameters
      var sy = 0.0
      var yy = 0.0
      forIndex(params.length)(i => {
        // difference in parameters
        oldParams(i) = {
          if (params(i).isInfinite && oldParams(i).isInfinite && (params(i) * oldParams(i) > 0)) 0.0
          else params(i) - oldParams(i)
        }
        // difference in gradients
        oldg(i) = {
          if (g(i).isInfinite && oldg(i).isInfinite && (g(i) * oldg(i) > 0)) 0.0
          else g(i) - oldg(i)
        }
        sy += oldParams(i) * oldg(i)
        yy += oldg(i) * oldg(i)
        direction(i) = g(i)
      })

      //      if (sy > 0.0 && sy < 0.001) {
      //        logger.info("Warning: sy=%f > 0".format(sy))
      //        sy = 0.0
      //      }
      if (sy > 0.0) {
        logger.info("Stepped over local maximum, resetting")
        //clear memory
        rho.clear()
        s.clear()
        y.clear()

        maths.set(oldParams, params)
        maths.set(oldg, g)

        //set direction
        direction *= 1.0 / direction.twoNorm
        step = 1.0
      } else {
        val gamma = sy / yy // scaling factor
        if (gamma > 0.0) throw new IllegalStateException("gamma=" + gamma + "> 0")

        pushDbl(rho, 1.0 / sy)
        pushArray(s, oldParams)
        pushArray(y, oldg)

        // calculate new direction
        assert(s.size == y.size)
        forReverseIndex(s.size)(i => {
          alpha(i) = rho(i) * direction.dot(s(i))
          direction.incr(y(i), -1.0 * alpha(i))
        })
        direction *= gamma
        forIndex(s.size)(i => {
          val beta = rho(i) * direction.dot(y(i))
          direction.incr(s(i), alpha(i) - beta)
        })

        forIndex(oldg.length)(i => {
          oldParams(i) = params(i)
          oldg(i) = g(i)
          direction(i) *= -1.0
        })
      }

      // take step in search direction
      step = lineMaximizer.optimize(direction, step)
      if (step == 0.0) {
        g = null
        step = 1.0
        logger.info("Line search could not step in the current direction. " +
          "(This is not necessarily cause for alarm. Sometimes this happens close to the maximum," +
          " where the function may be very flat.)")
        isConverged = true
        return true;
      }
      optimizable.getOptimizableParameters(params)
      optimizable.getOptimizableGradient(g)

      // after line search
      val newValue = optimizable.optimizableValue
      if (2.0 * math.abs(newValue - value) <= tolerance * (math.abs(newValue) + math.abs(value) + eps)) {
        logger.info("Exiting L-BFGS on termination #1:\nvalue difference below tolerance (oldValue: " + value + " newValue: " + newValue)
        isConverged = true
        return true;
      }
      val gg = g.twoNorm
      if (gg < gradientTolerance) {
        logger.info("Exiting L-BFGS on termination #2: \ngradient=" + gg + " < " + gradientTolerance)
        isConverged = true
        return true;
      }

      if (gg == 0.0) {
        logger.info("Exiting L-BFGS on termination #3: \ngradient==0.0")
        isConverged = true
        return true;
      }
      logger.info("Gradient = " + gg)
      iterations += 1
      if (iterations > maxIterations) {
        logger.info("Too many iterations in L-BFGS.java. Continuing with current parameters.")
        isConverged = true
        return true;
      }

      postIteration(iterationCount)
    })

    return false
  }
}
