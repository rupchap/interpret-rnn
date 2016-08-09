package edu.umass.cs.iesl.spdb

import io.Source
import collection.mutable.{ArrayBuffer, HashMap, HashSet}
import java.io.{File, PrintStream}
import edu.umass.cs.iesl.spdb.HumanAnnotation.Measure


/**
 * Created by IntelliJ IDEA.
 * User: lmyao
 * Date: 10/16/12
 * Time: 1:35 PM
 * To change this template use File | Settings | File Templates.
 */

object HumanAnnotation extends HasLogger{
  val dir =  "/iesl/canvas/lmyao/ie_ner/unsuprel/univSchema/"
  val testmentionfile = dir + "/nyt-freebase.test.triples.universal.mention.txt"  //from this file we extract entity pair -> sentence

  def main(args: Array[String]) {
    val subdir = if(args.length > 0) args(0) else "/iesl/canvas/lmyao/ie_ner/unsuprel/univSchema/9_10_2012/run_16_6_42_116"
    val pcaRankOutput = subdir  + "/nyt_pair.rank.txt"
    val output = pcaRankOutput + ".4annotation"
   // prepare4Annotation(pcaRankOutput,testmentionfile,output)

  //  rankByRelation("/iesl/canvas/lmyao/workspace/spdb/out/17_0_2013/run_16_31_53_730/nyt_pair.rank.eval.txt")    //mihai/mimlre-2012-06-13/corpora/multir/9_10_2012/test.tuples.rank.txt  ///17_10_2012/run_10_9_22_622/nyt_pair.rank.txt
  //  rankByRelation("/iesl/canvas/lmyao/workspace/spdb/out/24_0_2013/run_14_48_51_751/nyt_pair.rank.eval.txt")

    //signTest(dir + "/ds/nyt-freebase.test.universal.pred.txt", dir + "19_10_2012/run_15_20_57_450/nyt_pair.rank.txt", dir +"19_10_2012/me_gpca_sign.txt")

   // evalByAnnotation("/iesl/canvas/lmyao/workspace/spdb/out/26_1_2013/run_10_35_37_643/nyt_pair.rank.txt","/iesl/canvas/lmyao/ie_ner/unsuprel/univSchema/riedel_annotation/nyt-freebase.dev.sample.gold.tsv", "/iesl/canvas/lmyao/ie_ner/unsuprel/univSchema/riedel_annotation/nyt-freebase.dev.sample.pairs.txt", 0.1)
   evalByAnnotation(args(0), args(1), args(2) )
  }
  
  def prepare4Annotation(rankfile : String, sourcefile : String,  output : String){
    val pair2sen = new HashMap[String, HashSet[String]]    // arg1 -> rel arg1 arg2
    var source = Source.fromFile(sourcefile)
    for (line <- source.getLines; if (!line.startsWith("#Document"))) {
      val fields = line.split("\t")
      val pair = fields(1) + "\t" + fields(2)
      val sen = fields(fields.length -1)
      val sens = pair2sen.getOrElseUpdate(pair,new HashSet[String])
      sens += sen
    }
    source.close()

    val os = new PrintStream(output)
    // score  src dest  gold pred
    source = Source.fromFile(rankfile)
    for (instance <- source.getLines()){
      val fields = instance.split("\t")
      val pair = fields(1) + "\t" + fields(2)
      val sens : HashSet[String] = pair2sen.getOrElse(pair,HashSet.empty)
      if(fields(3).compareTo(fields(4)) == 0)   os.println("Y\t" + instance )
      else if(fields(3) != "NA")  os.println("N\t" + instance )
      else os.println ("\t" + instance )
      for(sen <- sens) {
        os.print("Sen:\t" + sen.replaceAll(fields(1),"["+fields(1)+"]").replaceAll(fields(2),"["+fields(2)+"]"))
      }
      
    }
    os.close
  }
  
  def annotate(mentionfile : String, labelfile : String, labelToAnnotate : String = "REL$/business/person/company"){
    val annotation= new HashMap[(Seq[Any],String),Boolean]
    for (line <- Source.fromFile(labelfile).getLines()) {
      val fields = line.split("\\t")
      val correct = fields(0) == "1"
      val label = fields(1)
      val tuple = fields.drop(2).toSeq
      annotation(tuple -> label) = correct
    }
    
    val os = new PrintStream(mentionfile+".4annotation")
    val source = Source.fromFile(mentionfile)
    for (line <- source.getLines(); if (!line.startsWith("UNLABELED"))){
      if (line.startsWith("#Document")) os.println(line)
      else{
        val fields = line.split("\t")
        val tuple = fields.drop(1).take(2).toSeq
        if (!annotation.contains(tuple->labelToAnnotate) ) {
          os.println("\t" +labelToAnnotate + "\t" + fields.drop(1).take(5).mkString ("\t") )
        }
      }
    }
    source.close
    os.close()
  }
  
  def rankByRelation(rankfile : String){
    val rel2list = new HashMap[String, ArrayBuffer[String]]
    val goldTotal = new HashMap[String, Int]
    val source = Source.fromFile(rankfile)
    for (line <- source.getLines) {
      val fields = line.split("\t")
      val rel = fields(fields.length-1)
      val goldrel = fields(fields.length-2)
      
      if(!goldTotal.contains(goldrel)) goldTotal += goldrel -> 1
      else goldTotal.update(goldrel,goldTotal(goldrel)+1)
      
      val list = rel2list.getOrElseUpdate(rel,new ArrayBuffer[String])
      list+= line
    }
    source.close
    
    var relsuffix = ""
    for ((rel,list)<-rel2list){
      relsuffix = rel.replaceAll("REL\\$","").substring(1).replaceAll("/","_")
      val os = new PrintStream(rankfile+"."+relsuffix+".rank")
      os.println(list.mkString("\n"))
      os.close
    }

    //output counts of gold instances for each label
    val os = new PrintStream(rankfile+".goldNum")
    //os.println(goldTotal.map(x => x._1 + "\t" + x._2).mkString("\n"))
    
    os.println("Cor\tGold\tTotal\tRec\tPrec")

    for ((rel,list)<-rel2list; if (goldTotal.contains(rel))){
      relsuffix = rel.replaceAll("REL\\$","").substring(1).replaceAll("/","_")
      val sos = new PrintStream(rankfile+"."+relsuffix+".curve")
      val scoreStr = eval4Relation(list,goldTotal(rel),sos)
      os.println(rel.replaceAll ("REL\\$","") + "\t" + scoreStr)
      sos.close
    }
    
    os.close
  }
  
  def eval4Relation(list : Seq[String], gold : Int,  os : PrintStream) : String = {
    var id = 0
    var cor = 0
    var total = 0
    for (instance <- list){
      val fields = instance.split("\t")
      if (fields(fields.length-1).compareTo(fields(fields.length-2)) == 0) {
        cor += 1
      }
      total += 1
      id += 1
      os.println(id + "\t" + cor * 1.0 / gold + "\t" + cor * 1.0 / total)
    }
    
    return cor + "\t" + gold + "\t" + total + "\t" +  cor * 1.0 / gold + "\t" +   cor * 1.0 / total
  }
  
  case class Measure() {
    var cor : Int = 0
    var gold : Int = 0
    var total : Int = 0
  }
  
  def evalByAnnotation(rankfile : String,  goldfile:String, tuplefile : String){
    val annotation= new HashSet[(Seq[Any],String)]
    val labels = new HashSet[String]
    val tuples = Util.loadHashSet(tuplefile)
    val numbers = new HashMap[String, Measure]
    val os = new PrintStream(rankfile+".sample.curve.txt")
    var gold = 0
    for (line <- Source.fromFile(goldfile).getLines()) {
      val fields = line.split("\\t")
      val label = fields(0)
      val tuple = fields.drop(1).toSeq
      if (tuples(tuple.mkString("\t"))) {
        annotation += tuple -> label
        labels += label
        val relNumbers = numbers.getOrElseUpdate(label, new Measure)
        relNumbers.gold += 1
        gold += 1
      }
    }
    println("Labels# " + labels.size)
    println("Tuples# " + tuples.size)
    println("Gold# " + gold)
    var AUC = 0.0
    var cor = 0
    var total = 0
    for (line <- Source.fromFile(rankfile).getLines()) {
      val fields = line.split("\t")
      val tuple = fields.drop(1).take(2).toSeq
      val pred = fields(fields.length-1)
      
      if (labels.contains(pred) && tuples(tuple.mkString("\t"))){
        val relNumbers = numbers.getOrElseUpdate(pred, new Measure)
        relNumbers.total += 1
        total += 1
        if (annotation.contains(tuple->pred)) {
          cor += 1
          relNumbers.cor += 1
        }
        val rec = cor * 1.0 / gold
        val prec = cor * 1.0 / total
        os.println(total + "\t" + rec + "\t" + prec)
        AUC += prec
      }

    }

    println("AUC: "+ AUC/total)

//    for ((rel,relNumbers) <- numbers){
//      os.println(rel + "\t" + relNumbers.cor + "\t" + relNumbers.gold + "\t" + relNumbers.total)
//    }
    
   // println("Sum\t"+cor+"\t"+gold+"\t"+total)
    os.close
  }
  
  //res1file: me  res2file:pca
  def signTest(res1file : String, res2file : String, output : String){
    val goldTriples2pred = new HashMap[String,String]()
    var source = Source.fromFile(res1file)
    for (line <- source.getLines()){
      val fields = line.split("\t")
      val triple = fields(1) + "\t" + fields(2)   //keep in mind that me format is different from pca
      goldTriples2pred += triple -> fields(fields.length-1)
    }
    source.close
    
    val os = new PrintStream(output)
    var c1 = 0
    var c2 = 0
    source = Source.fromFile(res2file)
    for (line <- source.getLines()){
      val fields = line.split("\t")
      val triple = fields(1)+"|"+fields(2)+"\t"+fields(3).replaceAll ("REL\\$","")
      val pred2 = fields(fields.length-1).replaceAll("REL\\$","")
      val pred1 = goldTriples2pred(triple)
      val goldrel = fields(3).replaceAll ("REL\\$","")
      if(pred1.compareTo(goldrel) == 0 && pred2.compareTo(goldrel)!= 0 ){
         os.println("1win\t" + triple + "\t" + pred1 + "\t" + pred2)
        c1+=1
      } else if (pred1.compareTo(goldrel) != 0 && pred2.compareTo(goldrel)== 0)  {
        os.println("2win\t" + triple + "\t" + pred1 + "\t" + pred2)
        c2+=1
      }
    }
    
    os.println("c1\tc2\t"+c1+"\t"+c2)
    os.close
  }

}


object SebAnnotate {
  def main(args: Array[String]) {
    //HumanAnnotation.prepare4Annotation(args(0),args(1),args(2))
    val mentionfile = if(args.length > 0) args(0) else "/iesl/canvas/lmyao/ie_ner/unsuprel/univSchema/riedel_annotation/nyt-freebase.dev.sample.1.universal.mention.txt"
    val labelfile = if(args.length > 1) args(1) else "/iesl/canvas/lmyao/ie_ner/unsuprel/univSchema/riedel_annotation/latest.tsv"
    val label = if(args.length > 2) args(2) else "REL$/book/author/works_written"
    HumanAnnotation.annotate(mentionfile,labelfile,label)
  }
}

