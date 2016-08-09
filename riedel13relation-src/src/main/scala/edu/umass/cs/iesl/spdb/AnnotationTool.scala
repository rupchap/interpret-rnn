package edu.umass.cs.iesl.spdb

import java.util.Calendar
import java.text.SimpleDateFormat
import java.io.{FileInputStream, InputStream, PrintStream, File}
import collection.mutable.{HashSet, HashMap}
import io.Source
import collection.mutable

/**
 * @author Sebastian Riedel
 */
object AnnotationTool {

  case class Annotation(tuple: Seq[Any], label: String, correct: Boolean) {
    override def toString = (Seq(if (correct) "1" else "0", label) ++ tuple).mkString("\t")

    def fact = tuple -> label
  }

  def loadAnnotations(in: InputStream, out: Option[PrintStream] = None) = {
    println("Reading in annotations...")
    val result = new mutable.HashMap[(Seq[Any], String), Annotation]()
    for (line <- Source.fromInputStream(in).getLines()) {
      val fields = line.split("\\t")
      val correct = fields(0) == "1"
      val label = fields(1)
      val tuple = fields.drop(2).toSeq
      result(tuple -> label) = Annotation(tuple, label, correct)
      for (o <- out) o.println(line)
    }
    result
  }

  def loadMentions(mentionFileName: String) = {
    val pair2sen = new HashMap[Seq[Any], HashSet[String]] // arg1 -> rel arg1 arg2
    val source = Source.fromFile(mentionFileName,"ISO-8859-1")
    println("Loading mention file...")
    for (line <- source.getLines(); if (!line.startsWith("#Document"))) {
      val fields = line.split("\t")
      val sen = fields(fields.length - 1)
      val sens = pair2sen.getOrElseUpdate(Seq(fields(1), fields(2)), new HashSet[String])
      sens += sen
    }
    source.close()
    pair2sen
  }


  def main(args: Array[String]) {

    val sourceName = args(0)
    val projDirName = args(1)
    val mentionFileName = args(2)
    val pattern = args.lift(3).getOrElse("").r
    val previousFileName = args.lift(4).getOrElse("latest.tsv")
    val newFileName = args.lift(5).getOrElse({
      val cal = Calendar.getInstance()
      val sdf = new SimpleDateFormat("yyyy-MM-dd-HH-mm-ss")
      sdf.format(cal.getTime) + ".tsv"
    })
    val projDir = new File(projDirName)
    projDir.mkdirs()

    val sourceFile = new File(sourceName)
    val previousFile = new File(projDir, previousFileName)
    val newFile = new File(projDir, newFileName)
    val out = new PrintStream(newFile)

    //read in mention file

    val pair2sen = loadMentions(mentionFileName)

    //read in previous file if exists
    //Format: Tuple, System,
    val annotations = if (previousFile.exists())
      loadAnnotations(new FileInputStream(previousFile), Some(out))
    else
      new mutable.HashMap[(Seq[Any], String), Annotation]
    println("Previous Annotations: " + annotations.size)

    //set up new softlink
    setupSoftlink(new File(projDir, "latest.tsv"), newFile)

    var labelled = 0

    //go through ranked file, and find tuples not yet annotated
    for (line <- Source.fromFile(sourceFile).getLines()) {
      val Array(score, arg1, arg2, freebase, predicted) = line.split("\\t")
      if (pattern.findFirstIn(predicted).isDefined) {
        val tuple = Seq(arg1, arg2)
        annotations.get(tuple -> predicted) match {
          case None =>
            //get sentences
            val sentences = pair2sen.getOrElse(tuple, Set.empty)
            //ask user
            println("*************************************************")
            println("Asking for annotation of: " + tuple.mkString(" | "))
            println("Number of annotations:    " + labelled)
            println("Prediction:               " + predicted)
            println("Score:                    " + score)
            println("Freebase:                 " + freebase)
            println("Sentences: ")
            for (sentence <- sentences) {
              var current = sentence
              for (arg <- tuple) {
                if (current.contains(arg)) {
                  current = current.replaceAll(arg, "[" + arg + "]")
                } else if (current.contains(arg.toUpperCase)) {
                  current = current.replaceAll(arg.toUpperCase, "[" + arg.toUpperCase + "]")
                } else if (current.contains(arg.toLowerCase)) {
                  current = current.replaceAll(arg.toLowerCase, "[" + arg.toLowerCase + "]")
                }
              }
              println("   " + current)
            }
            println("Correct (y/N)?: ")
            val line = readLine()
            val correct = line.trim.toLowerCase == "y"
            val annotation = Annotation(tuple, predicted, correct)
            out.println(annotation)
            out.flush()

          case Some(annotation) =>
        }
        labelled += 1
      }

    }


  }

  def setupSoftlink(latest: File, newFile: File) {
    if (latest.exists()) {
      //remove latest file, assuming it's a softlink
      latest.delete()
    }
    Runtime.getRuntime.exec("/bin/ln -s %s %s".format(newFile.getAbsolutePath, latest.getAbsolutePath))
  }
}
