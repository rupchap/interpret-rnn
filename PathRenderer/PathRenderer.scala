// package uclmr

// import java.io.PrintWriter

// import ml.wolfe.util.Conf
// import uclmr.io.LoadNAACL

import scala.io.Source
import scala.collection.mutable.ArrayBuffer
import java.io.File
import java.io.PrintWriter


/**
 * @author riedel
 */
object PathRenderer {

  def render(path: String, arg1: String = "A1", arg2: String  = "A2"): String = {
    // if(path.startsWith("REL")) {
    //   return s"$arg1${FreebaseStrings.phrase(path)}$arg2" //KB:
    // }
    val Array(_, middle, _) = path.split("\\|")
    val steps = middle.split("(?=->)|(?<=->)|(?=<-)|(?<=<-)").filter(_ != "")
    val inverse = path.endsWith("INV")
    val padded = if (inverse) (arg2 +: steps) :+ arg1 else (arg1 +: steps) :+ arg2

    //    def $-> = "->"
    //    def $<- = "<-"


    def renderSteps(remaining: List[String], start: List[String] = Nil, end: List[String] = Nil): List[String] = remaining match {
      case "<-" :: "nsubjpass" :: "<-" :: tail =>
        renderSteps(tail, start :+ "was", end) //need verb in past tense
      case "->" :: "nsubj" :: "->" :: tail =>
        renderSteps(tail, start, end) //check
      case "<-" :: "nsubj" :: "<-" :: tail =>
        renderSteps(tail, start, end)
      case "<-" :: "nn" :: "<-" :: tail =>
        renderSteps(tail, start, end)
      case "->" :: "dobj" :: "->" :: tail =>
        renderSteps(tail, start, end)
      case "->" :: "rcmod" :: "->" :: tail =>
        renderSteps(tail, start :+ "who", end)
      case "->" :: "partmod" :: "->" :: tail =>
        renderSteps(tail, start, end)
      case prep :: "->" :: "amod" :: "->" :: tail =>
        renderSteps(tail, start, end :+ prep)
      case "<-" :: "amod" :: "<-" :: tail =>
        renderSteps(tail, start :+ ",", end)
      case "->" :: "advmod" :: "->" :: tail =>
        renderSteps(tail, start :+ ",", end)
      case "<-" :: "appos" :: "<-" :: prep :: tail =>
        renderSteps(tail, prep :: start, end)
      case prep :: "->" :: "dep" :: "->" :: tail =>
        renderSteps(tail, start :+ prep, end)
      case "<-" :: "dep" :: "<-" :: tail =>
        renderSteps(tail, start, end)
      case prep :: "->" :: "pobj" :: "->" :: tail =>
        renderSteps(tail, start :+ prep, end)
      case "->" :: "iobj" :: "->" :: tail =>
        renderSteps(tail, start, end :+ "something")
      case "<-" :: "iobj" :: "<-" :: verb :: tail =>
        renderSteps(tail, "something" :: verb :: start, end)
      case "->" :: "prep" :: "->" :: prep :: "->" :: "pobj" :: "->" :: tail =>
        renderSteps(tail, start :+ prep, end)
      case "->" :: "prep" :: "->" :: dep :: "->" :: "dep" :: "->" :: prep :: "->" :: "pobj" :: "->" :: tail =>
        renderSteps(tail, start :+ dep :+ prep, end)
      case "<-" :: "pobj" :: "<-" :: prep :: "<-" :: "prep" :: "<-" :: word :: tail =>
        renderSteps(tail, word :: prep :: start, end)
      case "<-" :: "dobj" :: "<-" :: word :: tail =>
        renderSteps(tail, word :: start, end)
      case "->" :: "appos" :: "->" :: tail =>
        renderSteps(tail, start :+ ",", end)
      case "<-" :: "poss" :: "<-" :: tail =>
        renderSteps(tail, start :+ "'s", end)
      case word :: "->" :: "poss" :: "->" :: tail =>
        renderSteps(tail, start, "'s" :: word :: end)
      case word :: "->" :: "nn" :: "->" :: tail =>
        renderSteps(tail, start, word :: end)
      case "be" :: tail =>
        renderSteps(tail, start :+ "is", end)
//      case "_ARG1" :: tail =>
//        renderSteps(tail, start :+ arg1, end)
//      case "_ARG2" :: tail =>
//        renderSteps(tail, start :+ arg2, end)
      case head :: tail =>
        renderSteps(tail, start :+ head, end)
      case Nil => start ::: end
    }

    renderSteps(padded.toList).zipWithIndex.map(p => if(p._2==0) p._1 else {
      // if(p._1.startsWith(",") || p._1.startsWith("'")) p._1
      // else " " + p._1
      " " + p._1
    }).mkString("")

  }

  val examplePaths = Seq(
    "REL$/people/person/place_lived",
    "path#nsubjpass|<-nsubjpass<-challenge->prep->by->pobj->|pobj",
    "path#rcmod|->rcmod->be->nsubj->|nsubj:INV",
    "path#appos|->appos->nominee->amod->|amod",
    "path#appos|<-appos<-president->appos->|appos",
    "path#appos|->appos->freshman->dep->|dep",
    "path#nsubj|<-nsubj<-arrive->prep->in->pobj->|pobj",
    "path#appos|->appos->president->poss->|poss",
    "path#nsubj|<-nsubj<-be->prep->in->pobj->|pobj",
    "path#pobj|<-pobj<-in<-prep<-meeting->prep->with->pobj->|pobj",
    "path#partmod|->partmod->agree->prep->to->pobj->term->prep->with->pobj->|pobj",
    "path#appos|->appos->head->prep->of->pobj->|pobj",
    "rcmod|->rcmod->be->prep->in->pobj->|pobj",
    "nsubj|<-nsubj<-graduate->prep->of->pobj->|pobj",
    "dobj|<-dobj<-trade->prep->to->pobj->|pobj",
    "path#dobj|<-dobj<-play->prep->at->pobj->|pobj",
    "poss|<-poss<-company->nn->|nn",
    "path#poss|<-poss<-minister->appos->|appos:INV",
    "path#pobj|<-pobj<-from<-prep<-move->prep->to->pobj->|pobj:INV",
    "path#nsubj|<-nsubj<-name->dobj->president->nn->|nn:INV",
    "path#nn|<-nn<-production->prep->of->pobj->|pobj",
    "path#poss|<-poss<-invasion->prep->of->pobj->|pobj",
    "rcmod|->rcmod->coach->prep->at->pobj->|pobj",
    "path#dobj|<-dobj<-drive->prep->out->dep->of->pobj->|pobj",
    "path#amod|<-amod<-governor->prep->of->pobj->|pobj",
    "path#advmod|->advmod->back->dep->to->pobj->|pobj",
    "path#nsubj|<-nsubj<-award->iobj->|iobj",
    "path#iobj|<-iobj<-give->dobj->victory->prep->over->pobj->|pobj",
    "path#appos|<-appos<-director<-dep<-directed->partmod->release->prep->by->pobj->|pobj"
  )


  def main(args: Array[String]) {
  	val filename = "/data/train/paths.txt"
    var sentences = ArrayBuffer[String]()

  	for (line <- Source.fromFile(filename).getLines)
      sentences += render("path#" + line, "_ENTA", "_ENTB")

    val writer = new PrintWriter(new File("/data/train/shortsentences.txt"))
    for (line <- sentences)
      writer.println(line)
    writer.close()



  }

}

object FreebaseStrings {
  def phrase(rel: String) = rel match {
    case "REL$/people/person/parents" => " is a child of "
    case "REL$/film/film/written_by" => " was written by "
    case "REL$/book/written_work/subjects" => " is a book on "
    case "REL$/location/neighborhood/neighborhood_of" => " is a neighborhood of "
    case "REL$/business/person/company" => " is in "
    case "REL$/location/location/containedby" => " is inside "
    case "REL$/organization/parent/child" => " is the parent-company of "
    case "REL$/book/book_edition/author_editor" => " was edited by "
    case "REL$/people/deceased_person/place_of_death" => " died in "
    case "REL$/sports/sports_team/location" => " is based in "
    case "REL$/location/country/capital" => "'s capital is "
    case "REL$/people/person/place_lived" => " lived in "
    case "REL$/people/person/nationality" => "'s nationality is "
    case "REL$/book/author/book_editions_published" => " published a book on "
    case "REL$/people/person/place_of_birth" => " was born in "
    case "REL$/sports/sports_team/arena_stadium" => "'s home stadium is "
    case "REL$/film/film/directed_by" => " is directed by "
    case "REL$/sports/sports_facility/teams" => " is the home venue of "
    case "REL$/people/person/religion" => "'s religion is "
    case "REL$/broadcast/broadcast/area_served" => " is available in "
    case "REL$/book/author/works_written" => " wrote "
    case "REL$/location/country/administrative_divisions" => " contains "
    case "REL$/people/person/ethnicity" => "'s ethnicity is "
    case "REL$/book/book_edition/place_of_publication" => " is published in "
    case "REL$/business/company/place_founded" => " was founded in "
    case "REL$/music/artist/label" => " is under the label "
    case "REL$/location/hud_county_place/county" => " is in "
    case "REL$/aviation/airport/serves" => " serves "
    case "REL$/business/company/founders" => "'s founders include "
    case "REL$/visual_art/visual_artist/artworks" => " created "
    case "REL$/film/film/produced_by" => " was produced by "
    case "REL$/music/artist/album" => " released "
    case "REL$/location/us_county/hud_county_place" => " contains "
    case "REL$/sports/sports_team_owner/teams_owned" => " owns "
    case "REL$/media_common/netflix_genre/titles" => " is the genre of "
    case "REL$/architecture/structure/architect" => "'s architect is "
    case "REL$/music/artist/origin" => " originated in "
    case "REL$/film/film/country" => " originated in "
    case "REL$/music/record_label/artist" => " is a label with artist "
    case "REL$/business/company/major_shareholders" => "'s major shareholders include "
    case "REL$/location/us_county/county_seat" => "'s government is based in "
    case "REL$/architecture/architect/structures_designed" => " designed "
    case "REL$/military/military_person/participated_in_conflicts" => " participated in "
    case "REL$/sports/sports_team/league" => " is a team of "
    case "REL$/transportation/road/major_cities" => " goes through "
    case "REL$/fictional_universe/fictional_character/character_created_by" => " was created by "
    case "REL$/music/album/artist" => " is an album by "
    case "REL$/influence/influence_node/influenced_by" => " is influenced by "
    case "REL$/fictional_universe/fictional_character_creator/fictional_characters_created" => " created "
    case "REL$/music/composer/compositions" => " composed "
    case "REL$/sports/professional_sports_team/owner_s" => " is owned by "
  }
}
