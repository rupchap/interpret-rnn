package edu.umass.cs.iesl.spdb

import java.io.{FileOutputStream, File, PrintStream}


/**
 * @author sriedel
 */
trait HasLogger {

  lazy val logger = Logger

}


object Logger {
  val out = new PrintStream(new FileOutputStream(new File(Conf.outDir,"log.txt"),true))
  def info(msg: =>String) {
    val string = msg
    println(string)
    out.println(string)
  }
  def trace(msg: =>String) {
  }
}
