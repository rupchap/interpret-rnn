package edu.umass.cs.iesl.spdb

import java.io.{FileInputStream, File}
import collection.mutable
import io.Source

/**
 * @author Sebastian Riedel
 */
object ErrorAnalyzer {
  def main(args: Array[String]) {
    Conf.add(args(0))
    val mentions = AnnotationTool.loadMentions(args(1))
    val cooccurFile = new File(args(2))
    val result1File = new File(args(3))
    val result2File = new File(args(4))
    val pattern = args(5).r
    val goldFile = new File(Conf.conf.getString("eval.gold"))
    val annotations = AnnotationTool.loadAnnotations(new FileInputStream(goldFile))
    val cooccurCounts = new mutable.HashMap[(String, String), Int]()
    val poolDepth = Conf.conf.getInt("eval.pool-depth")
    val runDepth = Conf.conf.getInt("eval.run-depth")
    var currentTarget = ""
    for (line <- Source.fromFile(cooccurFile).getLines()) {
      if (!line.startsWith(" ")) currentTarget = line.trim
      else {
        val Array(count,currentSource) = line.trim.split(" ")
        cooccurCounts(currentSource -> currentTarget) = count.toInt
      }
    }
    val source = new mutable.HashMap[(Any,Any),Set[String]]
    println("Loading Source data.")
    for (line <- Source.fromFile(Conf.conf.getString("source-data.test")).getLines()) {
      val split = line.split("\t")
      val arg1 = split(1)
      val arg2 = split(2)
      source(arg1 -> arg2) = source.getOrElse(arg1->arg2,Set.empty) ++ split.drop(3)
    }


    val pool = new mutable.HashSet[(Any,Any,String)]()

    def loadResult(result:File) = {
      val positions = new mutable.HashMap[(Any,Any,String),Int]() {
        override def default(key: (Any, Any, String)) = runDepth + 1
      }
      var count = 0
      val lines = Source.fromFile(result).getLines()
      while (lines.hasNext && count < runDepth){
        val line = lines.next()
        val (arg1, arg2, predicted) = EvaluationTool.extractFactFromLine(line)
        if (pattern.findFirstIn(predicted).isDefined) {
          val tuple = Seq(arg1,arg2)
          val fact = tuple -> predicted
          positions((arg1,arg2,predicted)) = count
          if (count < poolDepth && annotations.get(fact).map(_.correct).getOrElse(false)){
            pool += ((arg1,arg2,predicted))
          }
          count += 1
        }
      }
      positions
    }
    val result1 = loadResult(result1File)
    val result2 = loadResult(result2File)

    val sorted = pool.toSeq.sortBy(fact => result1(fact) - result2(fact))

    println("Pool count: " + pool.size)

    for (fact@(arg1,arg2,relation) <- sorted) {
      val rank1 = result1(fact)
      val rank2 = result2(fact)
      val tuple = Seq(arg1,arg2)
      val sentences = mentions(tuple)
      val feats = source(arg1 -> arg2)
      println("============")
      println("Label: " + relation)
      println("Args:  %s | %s".format(arg1,arg2))
      println("Rank1: %d".format(rank1))
      println("Rank2: %d".format(rank2))
      for (feat <- feats.filter(_.startsWith("path"))) {
        val cooccurCount = cooccurCounts.getOrElse(feat -> relation, 0)
        println("  %4d %s".format(cooccurCount,feat))
      }
      for (sentence <- sentences) {
        println("  %s".format(sentence))
      }

    }

  }
}
