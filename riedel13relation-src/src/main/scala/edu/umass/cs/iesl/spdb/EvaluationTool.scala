package edu.umass.cs.iesl.spdb

import java.io.{FileInputStream, PrintStream, File}
import collection.JavaConversions._
import io.Source
import collection.mutable
import util.matching.Regex
import collection.mutable.ArrayBuffer
import org.sameersingh.scalaplot.{XYChart, MemXYSeries, XYSeries, XYData}
import org.sameersingh.scalaplot.gnuplot.GnuplotPlotter
import annotation.tailrec

/**
 * @author Sebastian Riedel
 */
object EvaluationTool {
  def main(args: Array[String]) {
    Conf.add(args(0))
    val rankFileNamesAndLabels = args.lift(1).getOrElse("out/latest/nyt_pair.rank.txt") +: args.drop(2)
    val rankFileNamesAndLabelsSplit = rankFileNamesAndLabels.map(name => if (name.contains(":")) name.split(":") else Array(name, new File(name).getName))
    val rankFileNames = rankFileNamesAndLabelsSplit.map(_.apply(0))
    val labels = rankFileNamesAndLabelsSplit.map(_.apply(1))
    val rankFiles = rankFileNames.map(new File(_))
    val goldFile = new File(Conf.conf.getString("eval.gold"))
    val relPatterns = Conf.conf.getStringList("eval.targets").toSeq.map(_.r)

//    evaluate(rankFiles, goldFile, new PrintStream("out/latest/eval.txt"), relPatterns, labels)
    evaluate(rankFiles, goldFile, System.out, relPatterns, labels)

  }

  class Eval(val pattern: Regex) {
    var totalGuess = 0

    def totalGoldTrue = goldTuplesTrue.size

    def totalGoldFalse = goldTuplesFalse.size

    var name: String = "N/A"

    def totalGold = totalGoldTrue + totalGoldFalse

    var tp = 0
    var fp = 0
    var sumPrecision = 0.0
    val precisions = new ArrayBuffer[Double]()
    val recalls = new ArrayBuffer[Double]()
    val missings = new ArrayBuffer[Int]()

    def interpolate(curve: Seq[(Double, Double)]) = {
      for (((r, p), index) <- curve.zipWithIndex) yield r -> curve.view.drop(index).map(_._2).max
    }

    def precisionRecallCurve(recallLevels: Seq[Double]) = {
      val p = 1.0 +: precisions
      val r = 0.0 +: recalls
      val result = new ArrayBuffer[(Double, Double)]
      var currentLevelIndex = 0
      def precAt(index: Int) = if (index == -1) 0.0 else p(index)
      for (level <- recallLevels) yield level -> precAt(r.indexWhere(_ >= level))
      //      var currentLevel = recallLevels(currentLevelIndex)
      //      var currentIndex = 0
      //      while (currentIndex < p.size && currentLevelIndex < recallLevels.size) {
      //        currentLevel = recallLevels(currentLevelIndex)
      //        val prec = p(currentIndex)
      //        val rec = r(currentIndex)
      //        if (rec >= currentLevel) {
      //          currentLevelIndex += 1
      //          //          result += rec -> prec
      //          result += currentLevel -> prec
      //
      //        }
      //        currentIndex += 1
      //      }
      //      result
    }

    var precisionCount = 0
    var mapDone = false
    val precisionAtK = new ArrayBuffer[(Int, Int, Double)]
    val avgPrecisionForFact = new mutable.HashMap[(Seq[Any], String), Double]
    val precisionForFact = new mutable.HashMap[(Seq[Any], String), Double]


    val relations = new mutable.HashSet[String]
    val goldTuplesTrue = new mutable.HashSet[(Seq[Any], String)]
    val goldTuplesFalse = new mutable.HashSet[(Seq[Any], String)]
    val guessTuplesTrue = new mutable.HashSet[(Seq[Any], String)]

    def copyGold = {
      val result = new Eval(pattern)
      result.relations ++= relations
      result.goldTuplesTrue ++= goldTuplesTrue
      result.goldTuplesFalse ++= goldTuplesFalse
      result.guessTuplesTrue ++= guessTuplesTrue
      result
    }

    def meanAvgPrecision = {
      var result = 0.0
      for (fact <- goldTuplesTrue) {
        val avgPrec = precisionForFact.getOrElse(fact, 0.0)
        result += avgPrec
      }
      result / goldTuplesTrue.size
    }

    def precisionAtRecall(recall: Double, depth: Int) = {
      if (recall == 0.0) 1.0
      else {
        val max = math.min(depth, precisions.size)
        val filtered = Range(0, max).filter(i => recalls(i) >= recall)
        if (filtered.size == 0) 0.0 else filtered.map(precisions(_)).max
      }
    }

    def precision = tp.toDouble / totalGuess

    def recall = tp.toDouble / totalGoldTrue

    def avgPrecision = sumPrecision / totalGuess

    def missingLabels = totalGuess - tp - fp

    override def toString = {
      """------------------
        |Pattern:       %s
        |Relations:     %s
        |Total Guess:   %d
        |Total Gold(T): %d
        |Total Gold(F): %d
        |True Pos:      %d
        |Precision:     %f
        |Recall:        %f
        |Avg Prec:      %f
        |Avg Prec#:     %d
        |MAP:           %f
        |Prec. at K:    %s""".stripMargin.format(pattern.toString(), relations.mkString(","),
        totalGuess, totalGoldTrue, totalGoldFalse, tp,
        precision, recall, avgPrecision, precisionCount, meanAvgPrecision, precisionAtK.mkString(", "))
    }
  }

  @tailrec
  def factorial(n: Double, result: Double = 1): Double = if (n == 0) result else factorial(n - 1, result * n)

  def nOverK(n: Double, k: Double) = factorial(n) / (factorial(k) * factorial(n - k))

  def binomial(k: Double, n: Double, p: Double) = nOverK(n, k) * math.pow(p, k) * math.pow(1 - p, n - k)

  def signtest(k: Int, n: Int) = {
    //p <= 2 * Sum (i=0 to k) {N!/(i!*(N-i)!)}/4
    //    val b = binomial(k,n,0.5)
    val sum = Range(0, k + 1).map(binomial(_, n, 0.5)).sum
    val result = 2.0 * sum
    result
    //    2.0 * b
    //    val nFact = factorial(n)
    //    val sum = Range(0,k+1).map( i => {
    //      val product = factorial(i) * factorial(n - i)
    //      nFact / product
    //    }).sum
    //    sum / 2.0
  }

  def evaluate(rankFiles: Seq[File], gold: File, out: PrintStream,
               relPatterns: Seq[Regex] = Conf.conf.getStringList("eval.targets").toSeq.map(_.r),
               names: Seq[String]) {
    val annotations = AnnotationTool.loadAnnotations(new FileInputStream(gold))
    val calculatePrecisionAtKs = Set(50, 100, 200, 300, 400)
    case class PerFileEvals(file: File, name: String, evals: mutable.HashMap[Regex, Eval] = new mutable.HashMap[Regex, Eval]()) {
      def averageMap() = evals.map(_._2.meanAvgPrecision).sum / evals.size

      def globalMap() = {
        val sum = evals.view.map(e => e._2.meanAvgPrecision * e._2.totalGoldTrue).sum
        val normalizer = evals.view.map(_._2.totalGoldTrue).sum
        sum / normalizer
      }

      def totalGoldTrue() = evals.view.map(_._2.totalGoldTrue).sum

      def averagePrecisionAt(recall: Double, depth: Int) = {
        evals.view.map(_._2.precisionAtRecall(recall, depth)).sum / evals.size
      }
    }
    val perFileEvals = new ArrayBuffer[PerFileEvals]
    val globalEvals = new mutable.HashMap[Regex, Eval]()

    val allowedFacts = new mutable.HashMap[Regex, mutable.HashSet[(Seq[Any], String)]]()
    val poolDepth = Conf.conf.getInt("eval.pool-depth")
    val runDepth = Conf.conf.getInt("eval.run-depth")
    val details = false


    println("Collecting facts from rank files")
    for (rankFile <- rankFiles) {
      val counts = new mutable.HashMap[Regex, Int]()
      val missing = new mutable.HashSet[Regex]()
      missing ++= relPatterns
      val lines = Source.fromFile(rankFile).getLines()
      while (lines.hasNext && !missing.isEmpty) {
        val line = lines.next()
        if (line.trim != "") {
          val (arg1, arg2, predicted) = extractFactFromLine(line)
          val tuple = Seq(arg1, arg2)
          val fact = tuple -> predicted

          for (pattern <- missing) {
            if (pattern.findFirstIn(predicted).isDefined) {
              allowedFacts.getOrElseUpdate(pattern, new mutable.HashSet[(Seq[Any], String)]()) += fact
              counts(pattern) = counts.getOrElse(pattern, 0) + 1
              if (counts(pattern) == poolDepth) missing -= pattern
            }
          }
        }
      }
    }

    println("Loading Annotations")
    for ((_, annotation) <- annotations) {
      for (pattern <- relPatterns; if (allowedFacts.get(pattern).map(_.apply(annotation.fact))).getOrElse(false)) {
        if (pattern.findFirstIn(annotation.label).isDefined) {
          val eval = globalEvals.getOrElseUpdate(pattern, new Eval(pattern))
          annotation.correct match {
            case true =>
              eval.goldTuplesTrue += annotation.tuple -> annotation.label
            case false =>
              eval.goldTuplesFalse += annotation.tuple -> annotation.label

          }
          eval.relations += annotation.label
        }
      }
    }

    println("Loading Rank Files")
    //todo: first make sure that for each pattern and system we are using at most K
    //todo: annotations from that system

    for ((rankFile, index) <- rankFiles.zipWithIndex) {
      val perFile = PerFileEvals(rankFile, names(index))
      import perFile._
      val counts = new mutable.HashMap[Regex, Int]()
      val missing = new mutable.HashSet[Regex]()
      missing ++= relPatterns
      evals ++= globalEvals.mapValues(_.copyGold)
      val lines = Source.fromFile(rankFile).getLines()
      while (lines.hasNext && !missing.isEmpty) {
        val line = lines.next()
        val (arg1, arg2, predicted) = extractFactFromLine(line)
        val tuple = Seq(arg1, arg2)
        val fact = tuple -> predicted
        for (pattern <- relPatterns) {
          val eval = evals.getOrElseUpdate(pattern, new Eval(pattern))
          if (pattern.findFirstIn(predicted).isDefined) {
            eval.relations += predicted
            eval.totalGuess += 1
            eval.goldTuplesTrue(fact) -> eval.goldTuplesFalse(fact) match {
              case (true, _) =>
                eval.tp += 1
                eval.guessTuplesTrue += fact

              case (false, true) =>
                eval.fp += 1
              case (false, false) =>
            }
            eval.sumPrecision += eval.precision
            eval.precisions += eval.precision
            eval.recalls += eval.recall
            eval.missings += eval.missingLabels
            if (eval.goldTuplesTrue(fact)) {
              eval.avgPrecisionForFact(fact) = eval.avgPrecision
              eval.precisionForFact(fact) = eval.precision
            }
            if (calculatePrecisionAtKs(eval.totalGuess)) {
              eval.precisionAtK += ((eval.totalGuess, eval.missingLabels, eval.precision))
            }
            counts(pattern) = counts.getOrElse(pattern, 0) + 1
            if (counts(pattern) == runDepth) missing -= pattern

          }
        }
      }
      for (pattern <- relPatterns; eval <- evals.get(pattern)) {
        if (details) out.println(eval)
      }
      perFileEvals += perFile
    }


    //print overview table
    def printTextTable() {
      out.println("Summary:")
      out.print("%-30s%-10s%-10s".format("Pattern", "Gold+", "Gold+-"))
      for ((perFile, index) <- perFileEvals.zipWithIndex) {
        out.print("| %-10s%-10s".format("MAP", "Missing"))
      }
      out.println()
      out.print("%50s".format(Range(0, 50).map(s => "-").mkString))
      for (perFile <- perFileEvals) {
        out.print("%22s".format(Range(0, 22).map(s => "-").mkString))
      }
      out.println()
      for (pattern <- relPatterns.sortBy(pattern => -perFileEvals.head.evals(pattern).totalGoldTrue)) {
        val first = perFileEvals.head
        out.print("%-30s%-10d%-10d".format(pattern.toString(), first.evals(pattern).goldTuplesTrue.size, first.evals(pattern).totalGold))
        for (perFile <- perFileEvals) {
          val eval = perFile.evals(pattern)
          out.print("| %-10.2f%-10d".format(
            eval.meanAvgPrecision,
            //          eval.precisionAtK.lift(1).map(_._3).getOrElse(-1.0)
            eval.missings.lift(math.min(poolDepth, eval.missings.size) - 1).getOrElse(-1)
          ))
        }
        out.println()
      }
      out.print("%-30s%-10d%-10d".format("Average", 0, 0))
      for (perFile <- perFileEvals) {
        out.print("| %-10.2f%-10d".format(perFile.averageMap(), -1))
      }
      out.println()
      out.print("%-30s%-10d%-10d".format("Global", 0, 0))
      for (perFile <- perFileEvals) {
        out.print("| %-10.2f%-10d".format(perFile.globalMap(), -1))
      }
      out.println()
    }

    //print latex table
    def printLatexTable() {
      def norm(label: String) = label.replaceAll("\\$", "").replaceAll("_", "\\\\_")
      val systemCount = perFileEvals.size
      out.println("Latex:")
      out.println("\\begin{center}")
      out.println("\\begin{tabular}{ %s %s | %s }".format("l", "l", Seq.fill(systemCount)("c").mkString(" ")))
      out.println("  %20s & %s & %s \\\\".format("Relation", "\\#", perFileEvals.map(_.name).mkString(" & ")))
      out.println("\\hline")
      for (pattern <- relPatterns.sortBy(pattern => -perFileEvals.head.evals(pattern).totalGoldTrue)) {
        val first = perFileEvals.head
        val maps = perFileEvals.map(_.evals(pattern).meanAvgPrecision)
        val sorted = maps.sortBy(-_)
        def format(map: Double) = map match {
          case x if (x >= sorted.head && x <= sorted(1)) => "{\\em %6.2f}".format(map)
          case x if (x >= sorted.head) => "{\\bf %6.2f}".format(map)
          case _ => "%6.2f".format(map)
        }


        out.println("  %20s & %4d & %s \\\\".format(norm(pattern.toString()), first.evals(pattern).totalGoldTrue,
          maps.map(format(_)).mkString(" & ")))
      }
      out.println("\\hline")
      out.println("  %20s & %4s & %s \\\\".format("MAP",
        "",
        perFileEvals.map(e => "%6.2f".format(e.averageMap())).mkString(" & ")))
//      out.println("\\hline")
      out.println("  %20s & %4s & %s \\\\".format("Weighted MAP",
        "",
        perFileEvals.map(e => "%6.2f".format(e.globalMap())).mkString(" & ")))
      out.println("\\end{tabular}")
      out.println("\\end{center}")
    }

    printLatexTable()
    printTextTable()




    //print pairwise comparisons
    out.println(("name" +: perFileEvals.map(_.name)).map(title => "%-13s".format(title)).mkString)
    for (i1 <- 0 until perFileEvals.size; i2 <- i1 + 1 until perFileEvals.size;
         pf1 = perFileEvals(i1)) {
      val cells = for (i2 <- i1 + 1 until perFileEvals.size; pf2 = perFileEvals(i2)) yield {
        var wins = 0
        var total = 0
        for (pattern <- relPatterns) {
          val eval1 = pf1.evals(pattern)
          val eval2 = pf2.evals(pattern)
          if (math.abs(eval1.meanAvgPrecision - eval2.meanAvgPrecision) > 0.001) {
            val win = eval1.meanAvgPrecision > eval2.meanAvgPrecision
            if (win) wins += 1
            total += 1
          }
        }
        val losses = total - wins

        val pValue = signtest(math.min(wins, losses), total)
        "%2d/%2d %5.3f".format(wins, losses, pValue)
      }
      out.println((pf1.name +: (Range(0, i1 + 1).map(i => "") ++ cells)).map("%-13s".format(_)).mkString)

    }

  /*  //print graph
    val evalDir = new File("eval")
    evalDir.mkdirs()
    for (pattern <- relPatterns) {
      val data_recallPrec = new XYData()
      val data_precAt = new XYData()

      val first = perFileEvals.head
      val x = Range(0, runDepth).map(_.toDouble)
      for (perFile <- perFileEvals) {
        val eval = perFile.evals(pattern)
        val raw = eval.precisionRecallCurve(Range(0, 50).map(_ / 50.0))
        val curve = eval.interpolate(raw)
        val curve_x = curve.map(_._1)
        val curve_y = curve.map(_._2)
        val y = eval.precisions.take(runDepth)
        val series = new MemXYSeries(curve_x, curve_y, perFile.name)
        val series_precAt = new MemXYSeries(x.take(y.length), y, perFile.name)
        data_recallPrec += series
        data_precAt += series_precAt
      }
      val chart = new XYChart("Precision at K", data_precAt)
      chart.showLegend = true
      chart.xlabel = "Facts"
      chart.ylabel = "Precision"
      val plotter = new GnuplotPlotter(chart)
      plotter.writeToPdf("eval/", pattern.toString().replaceAll("/", "_"))

      val chartRecallPrec = new XYChart("Recall/Precision", data_recallPrec)
      chartRecallPrec.showLegend = true
      chartRecallPrec.xlabel = "Recall"
      chartRecallPrec.ylabel = "Precision"
      val plotterRecallPrec = new GnuplotPlotter(chartRecallPrec)
      plotterRecallPrec.writeToPdf("eval/", pattern.toString().replaceAll("/", "_") + "-rp")

    }  */

    //print 11 point avg precision graph
    {
      val data_recallPrec = new XYData()
      val recalls = Seq(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
      for (perFile <- perFileEvals) {
        val series = new MemXYSeries(recalls, recalls.map(r => perFile.averagePrecisionAt(r, runDepth)), perFile.name)
        data_recallPrec += series
      }
      val chartRecallPrec = new XYChart("Averaged 11-point Precision/Recall", data_recallPrec)
      chartRecallPrec.showLegend = true
      chartRecallPrec.xlabel = "Recall"
      chartRecallPrec.ylabel = "Precision"
      val plotterRecallPrec = new GnuplotPlotter(chartRecallPrec)
      plotterRecallPrec.writeToPdf("eval/", "11pointPrecRecall")
    }




    //    out.println("Missing:")
    //    for (pattern <- relPatterns) {
    //      val eval = evals(pattern)
    //      val missing = eval.goldTuplesTrue -- eval.guessTuplesTrue
    //      if (!missing.isEmpty) {
    //        out.println("--------")
    //        out.println("Regex: " + pattern)
    //        out.println(missing.mkString(","))
    //      }
    //    }


  }


  def extractFactFromLine(line: String): (String, String, String) = {
    val split = line.split("\\t")
    if (split.size == 4) {
      val Array(arg1, arg2) = split(1).split("\\|")
      (arg1, arg2, split(3))
    } else {
      (split(1), split(2), split(4))
    }
  }

  def extractScoredFactFromLine(line: String): (Double, String, String, String) = {
    val split = line.split("\\t")
    if (split.size == 4) {
      val Array(arg1, arg2) = split(1).split("\\|")
      (split(0).toDouble, arg1, arg2, split(3))
    } else {
      (split(0).toDouble, split(1), split(2), split(4))
    }
  }

}

object FilterRankFile {
  def main(args: Array[String]) {
    val source = args(0)
    val filterTuple = args(1)
    val dest = args(2)
    filter(dest, filterTuple, source)

  }

  def filter(dest: String, filterTuple: String, source: String) {
    val allowed = new mutable.HashSet[Seq[Any]]()

    val out = new PrintStream(dest)

    for (line <- Source.fromFile(filterTuple).getLines(); if (line.trim != "")) {
      val split = line.split("\t")
      val tuple = if (split.size==2) Seq(split(0),split(1)) else Seq(split(1), split(2))
      allowed += tuple
    }
    println(allowed.size)

    def norm(label: String) = if (label.contains("/") && !label.startsWith("REL$")) "REL$" + label else label

    for (line <- Source.fromFile(source).getLines()) {
      val split = line.split("[\t]")
      if (split(1).contains("|")) {
        val tuple = split(1).split("\\|").toSeq
        if (allowed(tuple)) out.println(split(0) + "\t" + tuple.mkString("\t") + "\t" + split.drop(2).map(norm).mkString("\t"))
      } else {
        val tuple = Seq(split(1), split(2))
        if (allowed(tuple)) out.println(split.take(3).mkString("\t") + "\t" + split.drop(3).map(norm).mkString("\t"))
      }
    }

    out.close()
  }
}

object MihaiConverter {
  def main(args: Array[String]) {
    val source = args(0)
    val dest = args(1)
    val out = new PrintStream(dest)

    val buffer = new ArrayBuffer[(String, Double)]()

    def pos(value: Double) = value //if (value > 0.0) value else - value

    for (line <- Source.fromFile(source).getLines(); if (line.trim != "")) {
      val split = line.split("\t")
      val arg1 = split(0)
      val arg2 = split(1)
      val freebase = split(3)
      val indexOfPreds = split.indexOf("pred")
      if (split(indexOfPreds + 1) != "NA") {
        val label1 = split(indexOfPreds + 1)
        val score1 = pos(split(indexOfPreds + 2).toDouble)
        buffer += Seq(score1, arg1, arg2, freebase, label1).mkString("\t") -> score1
      }
      if (split.length > indexOfPreds + 3) {
        val label2 = split(indexOfPreds + 3)
        val score2 = pos(split(indexOfPreds + 4).toDouble)
        buffer += Seq(score2, arg1, arg2, freebase, label2).mkString("\t") -> score2

      }
    }

    val sorted = buffer.sortBy(-_._2)

    for ((line, score) <- sorted) {
      out.println(line)
    }

    out.close()

  }
}
