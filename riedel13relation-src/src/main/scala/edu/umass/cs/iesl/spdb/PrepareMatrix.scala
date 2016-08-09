package edu.umass.cs.iesl.spdb

import io.Source
import cc.factorie.CategoricalDomain
import collection.mutable.{ArrayBuffer, HashSet, HashMap}
import java.util.ArrayList
import java.io._

/**
 * Created by IntelliJ IDEA.
 * User: lmyao
 * Date: 5/3/12
 * Time: 2:15 PM
 * To change this template use File | Settings | File Templates.
 */

object PrepareMatrix {
  val CUTTING = 2
  val CUTTING_ROW = 1
  val NEGATIVE = 5
  object FeatureDomain extends CategoricalDomain[String]
  def main(args: Array[String]){
    val dir = "/iesl/canvas/lmyao/ie_ner/unsuprel/tackbp"
    val subdir = "/19-06-2012"
  //  groupByPairs(dir + subdir + "/tac_link.txt",  dir + subdir +  "/tac_pair")

    val pairfile = dir + subdir + "/tac_pair.rowid.txt"
    val dictfile = dir + subdir + "/tac_pair.dict.txt"
    val betafile = dir + subdir +  "/50/tac_pair.model.beta.txt"
    val zfile = dir + subdir + "/50/tac_pair.pred.z.txt"
    val predfile = dir + subdir + "/50/tac_pair.pred.x.txt"
    val predfileData = dir + subdir + "/50/tac_pair.reconstruction.txt"

    //only consider test rows
   // predict(zfile, betafile, predfile, predfileData, pairfile, dictfile, 6439,  102)

    val goldfile = dir + "/tac_test_relations.txt"
    val errfile = dir + subdir + "/errpred.txt"
    evaluate(predfileData,goldfile,errfile)
  }

  //create feature-label features, group, get feature index, matrix row, column, heldout matrix
  def groupByPairs(mentionPairfile : String, output : String){
    val matOut = output + ".dat"
    val heldOut = output + ".heldout.dat"
    val trainOut = output + ".train"
    val testOut = output + ".test"
    val trainos = new PrintStream(trainOut)
    val testos = new PrintStream(testOut)

    val taclabels = new HashSet[String]()

    val pairMap = new HashMap[Int,String]    //index starts with 0

    val feastat = new HashMap[String, Int] // for sampling negative features
    val featureSeq = new ArrayBuffer[Int]
    
    val trainMap = new HashMap[String, HashSet[String]]
    val testMap = new HashMap[String, HashSet[String]]()
    var source = Source.fromFile(mentionPairfile)
    for(line <- source.getLines() ; if(!(line.startsWith("#Document"))  && !(line.startsWith("other")) )){          //
      val fields = line.split("\t")
    if (fields.size > 1){
      var pair = fields(1)
      var nertype = ""
      var features : HashSet[String] = null
      if(fields(0).startsWith( "train") /*|| fields(0) == "other"*/){
        features = trainMap.getOrElseUpdate(pair,new HashSet[String])
      }
      else if (fields(0).startsWith( "test")) {
        features = testMap.getOrElseUpdate(pair,new HashSet[String]())
      }
      for(field <- fields.drop(2); if(field.startsWith("lex") || field.startsWith("pos") || field.startsWith("type#") )){
        if(field.startsWith("type#")){
          nertype = field.replace("type#","")
        } else{
          features += field
          if(feastat.contains(field)) feastat.update(field,feastat(field)+1)
          else feastat += field -> 1
        }
      }
      if(fields(0).startsWith("train")){
        val label = if(fields(0).contains( ":") ) fields(0).replace("train:","")  else ""
        if(label.length>0){
          features += label
          taclabels += label
          FeatureDomain += label
          if(nertype.compareTo("NONE->MISC")!= 0 && nertype.compareTo("NONE->NONE") != 0 && nertype.compareTo("MISC->NONE")!= 0 )
          features += label+"|"+nertype
          FeatureDomain += label+"|"+nertype
        }
      }
    }
    }

    var rowid = 0
    val os = new PrintStream(matOut)
    val hos = new PrintStream(heldOut)
    for(entstr <- trainMap.filter(_._2.size > 0).keys){

      val features = trainMap(entstr)
      val indices = new ArrayBuffer[Int]
      for(feature <- features){
        if(FeatureDomain.contains(feature) ||  feastat(feature) >= CUTTING){
          FeatureDomain += feature
          indices += FeatureDomain.getIndex(feature)
          if(!taclabels.contains(feature))  featureSeq += FeatureDomain.getIndex(feature)
        }
      }
      if(indices.size >= CUTTING_ROW)  {   //CUTTING_ROW
        pairMap += rowid -> entstr
        rowid += 1
        for(fid <- indices){
          os.println(rowid + "\t" + (fid+1) + "\t1")
          hos.println(rowid + "\t" + (fid+1) + "\t1")
        }
        trainos.println(entstr + "\t" + indices.map(x => FeatureDomain.getCategory(x)).mkString("\t"))
      }
    }
    println("Train rows: " + rowid)
    for(entstr <- testMap.filter(_._2.size > 0).keys){    //test has no labels for each pair, we want to predict labels based on these observations, now we assume we know what the candidates are
      val features = testMap(entstr)
      val indices = new ArrayBuffer[Int]
      for(feature <- features){
        if(feastat(feature) >= CUTTING ){
          FeatureDomain += feature
          indices += FeatureDomain.getIndex(feature)
          featureSeq += FeatureDomain.getIndex (feature)
        }
      }
      if(indices.size >= CUTTING_ROW)  {
        pairMap += rowid -> entstr
        rowid += 1
        for(fid <- indices){
          os.println(rowid + "\t" + (fid+1) + "\t1")
          hos.println(rowid + "\t" + (fid+1) + "\t1")
        }
        testos.println(entstr + "\t" + indices.map(x => FeatureDomain.getCategory(x)).mkString("\t"))
      }
    }
    trainos.close()
    testos.close

    //generate negative features now,  should exclude positive features,
    rowid = 0
    source = Source.fromFile(trainOut)
    for(line <- source.getLines()){
      rowid += 1
      val features = new HashSet[Int]
      line.split("\t").drop(1).map(x => FeatureDomain.getIndex(x)).map(x => features += x)

      val size = features.size
      var negsize = size * NEGATIVE
      while(negsize > FeatureDomain.size - size)  negsize = negsize/2
      
      for(iter <- 0 until negsize){
        var index = (math.random * featureSeq.size).toInt
        var fid = featureSeq(index)
        while(features.contains(fid)) {
          index += 1
          index = index % featureSeq.size
          fid = featureSeq(index)
        }
        hos.println(rowid + "\t" + (fid+1) + "\t1")
      }

      //add all other labels as negative features for training data , only for training data
      for(label <- taclabels){
        val fid = FeatureDomain.getIndex(label)
        if(!features.contains(fid)) hos.println(rowid + "\t" + (fid+1) + "\t1")
      }
    }

    source = Source.fromFile(testOut)
    for(line <- source.getLines()){
      rowid += 1
      val features = new HashSet[Int]
      line.split("\t").drop(1).map(x => FeatureDomain.getIndex(x)).map(x => features += x)

      val size = features.size
      var negsize = size * NEGATIVE
      while(negsize > FeatureDomain.size - size)  negsize = negsize/2

      for(iter <- 0 until negsize){
        var index = (math.random * featureSeq.size).toInt
        var fid = featureSeq(index)
        while(features.contains(fid)) {
          index += 1
          index = index % featureSeq.size
          fid = featureSeq(index)
        }
        hos.println(rowid + "\t" + (fid+1) + "\t1")
      }
    }

    os.println(rowid + "\t" + FeatureDomain.size + "\t0")


    hos.println(rowid + "\t" + FeatureDomain.size + "\t0")
    os.close
    hos.close
    
    println("Total features: " + FeatureDomain.size)
    val dictfile = output + ".dict.txt"
    var dict = new PrintStream(dictfile)
    for(index <- 0 until FeatureDomain.size)
      dict.println(FeatureDomain.getCategory(index) + "\t" + index)
    dict.close()
    
    val pairfile = output + ".rowid.txt"
    dict = new PrintStream(pairfile)
    dict.println(pairMap.map(x => x._1 + "\t" + x._2).mkString("\n"))
    dict.close()

    //feature statistics
    dict = new PrintStream(output + ".stat.txt")
    dict.println(feastat.filter(_._2 >= CUTTING).map(x => x._2 + "\t" + x._1).mkString("\n"))
    dict.close()
  }

  //we care about the label features, with fid < 38 , numTopFeas : additional features besides the labels , todo: map back to string labels, row->pair, FeatureDomain
  def predict(zfile : String, betafile : String, predfile : String, predfileData : String , pairfile : String, dictfile : String,  rowidStart: Int, colidEnd : Int){
    val dictMap = Util.loadRevHash(dictfile)

    val zmap = new HashMap[String, Double]
    val betamap = new HashMap[String, Double]
    var rows = -1
    var cols = -1
    var K = -1
    var threshold = 0

    var x = loadMap(zmap,zfile)
    rows = x._1
    K = x._2
    x= loadMap(betamap,betafile)
    cols = x._2
    
    val pairMap = Util.loadHash(pairfile)
    val mos = new PrintStream(predfile)
    val os = new PrintStream(predfileData)
    val startTime = System.currentTimeMillis
    for( i <- rowidStart until rows){
      val tmptable = new HashMap[Int,Double]
      for(j <- 0 until colidEnd){
        var tmpval = 0.0
        for(k <- 0 until K){
          tmpval += zmap(i+"\t"+k)*betamap(k+"\t"+j)
        }
        if(tmpval > threshold) {
          tmpval = 1.0/ (1+ math.exp(-tmpval))
          tmptable += j -> tmpval
        }
      }
      os.print(pairMap(i.toString) + "\t")
      os.println( tmptable.toList.sortWith((x,y) =>  x._2 > y._2).map({ case x =>  dictMap(x._1.toString)  + ":" + x._2 } ).mkString("\t")  )
      mos.println( tmptable.toList.sortWith((x,y) =>  x._2 > y._2).map({ case x => i + "\t" +  x._1  + "\t" + x._2 } ).mkString("\n") )
    }
    println("Finished in " + ((System.currentTimeMillis - startTime) / 1000.0) + " seconds")
    os.close
    mos.close
  }

  def loadMap(table : HashMap[String, Double], file : String) : (Int,Int) = {
    val fstream = new FileInputStream(file)
    // Get the object of DataInputStream
    val in = new DataInputStream(fstream)
    val br = new BufferedReader(new InputStreamReader(in))
    var cols = -1
    var strLine : String = ""
    var lineno = 0
    strLine = br.readLine

    //Read File Line By Line
    while(strLine != null){
      val fields = strLine.split("\t")
      cols = fields.length
      for(i <- 0 until fields.length){
        val key = lineno + "\t" + i
        table += key -> fields(i).toDouble
      }
      lineno += 1
      strLine = br.readLine()
    }
    in.close
    (lineno,cols)
  }
  
  def evaluate(predfile : String,  goldfile : String, errfile : String){
    var goldcount = 0
    var predcount = 0
    var corcount = 0
    val os = new PrintStream(errfile)
    val goldTuples = Util.loadHashSetLowercase(goldfile)
    goldcount = goldTuples.size
    val source = Source.fromFile(predfile)
    for (line <- source.getLines ) {
      val fields = line.split("\t")
      val pair = fields(0).split("\\|").mkString("\t").toLowerCase
      val pairRev = fields(0).split("\\|").reverse.mkString("\t").toLowerCase
      val predLabels = new HashSet[String]()
      for( field <- fields.drop(1)){
        var pred = field
        if(field.matches("[a-z]+:[a-z]+.*")){
          val splitpos = if(field.contains("|")) field.indexOf("|") else field.lastIndexOf(":")
          pred = pred.substring(0,splitpos)
          predLabels += pred
        }
      }
      for( label <- predLabels){
        if(goldTuples.contains(label + "\t" + pair)){
          os.println("Cor\t" + pair + "\t" +  label)
          corcount += 1
        } else if(goldTuples.contains(label + "\t" + pairRev)){
          os.println("Cor\t" + pairRev + "\t" +  label)
          corcount += 1
        }
        else
          os.println(pair + "\t" + label)
        predcount += 1

      }

    }
    
    println("cor\tpred\tgold")
    println(corcount + "\t" + predcount + "\t" + goldcount)
    os.close()
  }

}