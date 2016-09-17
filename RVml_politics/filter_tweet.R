library(jsonlite)
library (stringr)

args=(commandArgs(TRUE))

##args is now a list of character vectors
## First check to see if arguments are passed.
## Then cycle through each element of the list and evaluate the expressions.
if(length(args)==0){
  
  print("No arguments supplied.")
  
}else{
  unCleanJSON = args[1]
#  res = readLines (unCleanJSON);
#  if (validate (res[1]) == TRUE)
#  {
   #  unCleanJSON = read.table (args[1], check.names=F, sep="", col.names=F, stringsAsFactors=F, 
     json_file = fromJSON(unCleanJSON, flatten = TRUE)
     jsonText = data.frame(text=json_file$text)
     jsonText$text = str_replace_all(jsonText$text, "#", "")
     jsonText$text = str_replace_all(jsonText$text, "\n", " ")
     jsonText$text = str_replace_all(jsonText$text, "\"", "")
     jsonText$text = gsub("@\\S+\\s*","", jsonText$text,ignore.case=T)
     jsonText$text = gsub("http\\S+\\s*","", jsonText$text,ignore.case=T)
     jsonText$text = gsub("^.*?RT ","", jsonText$text,ignore.case=T) 

     location=json_file$retweeted_status$place$bounding_box$coordinates
     len=length(location)
     long=location[1]
     lat =location[(len/2)+1]

     if (is.null (long))
     {
      long = "0";
     }
     if (is.null (lat))
     {
      lat = "0";
     }

#     blah = sprintf ("blah: %s\n%s\n%s\n%s", jsonText$text, as.character (NULL), "d", "c");
     blah = sprintf ("%s\n%s\n%s", jsonText$text, as.character (long), as.character (lat));
     cat (blah);

#  }
#  else
#  {
#   print ("Invalid JSON detected, skipping");
#  }
}
