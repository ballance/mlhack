require(maps)
library(jsonlite)
library(stringr)

#gpclibPermit()
library(gpclib)
library(RColorBrewer) # creates nice color schemes
library(reshape)
library(classInt)

testit <- function(x)
{
  p1 <- proc.time()
  Sys.sleep(x)
  proc.time() - p1 # The cpu usage should be negligible
}

setwd("/Users/pal004/PoliticPredictors")

finf = file.info(dir(), extra_cols = FALSE)
rowIndex = which(row.names(finf) == "FALSE")
finf=finf[1:rowIndex-1,]

if(any(row.names(finf) %in% "StateFile.csv")){
  
    stateMetrics = read.csv("StateFile.csv")
 
while (1==1) 
{
  print ("while loop 1");
  if(any(row.names(finf) %in% "output.json")){
    
    jsonRow=finf[which(row.names(finf)=="output.json"),]
    maxTime = data.frame(dateTime=jsonRow$mtime)
    oldTime = maxTime
    #fileName = row.names(jsonRow)
    twitData = read.table("output.json", sep="\n", comment.char = "#")
    lenTwit = length(twitData[,1])
    
    print (lenTwit);
    if(lenTwit == 5 && twitData[lenTwit,] == 'EOL'){
    
    while(maxTime$dateTime <= Sys.time() & maxTime$dateTime <= oldTime$dateTime){
      print ("while loop 2");
      
      finf = file.info(dir(), extra_cols = FALSE)
      rowIndex = which(row.names(finf) == "FALSE")
      finf=finf[1:rowIndex-1,]
      
      jsonRow=finf[which(row.names(finf)=="output.json"),]
      maxTime = data.frame(dateTime=jsonRow$mtime)
      
      #json_file = fromJSON(eval(fileName), flatten = TRUE)
      
      #jsonText = data.frame(text=json_file$text)
      #jsonText$text = str_replace_all(jsonText$text, "#", "")
      #jsonText$text = str_replace_all(jsonText$text, "\n", " ")
      #jsonText$text = str_replace_all(jsonText$text, "\"", "")
      #jsonText$text = gsub("@\\S+\\s*","", jsonText$text,ignore.case=T)
      #jsonText$text = gsub("http\\S+\\s*","", jsonText$text,ignore.case=T)
      #jsonText$text = gsub("^.*?RT ","", jsonText$text,ignore.case=T)
      
        #location=json_file$retweeted_status$place$bounding_box$coordinates
        prob = runif(1,0,1)
        index = sample(1:64,1,replace=T)
        
        currSize = stateMetrics[index,]$Size
        currCount = stateMetrics[index,]$Count
        stateMetrics[index,]$Size = (currCount * currSize + prob) / (currCount + 1)
        stateMetrics[index,]$Count = currCount + 1
        
        write.csv(stateMetrics, "StateFile.csv", row.names=FALSE)
      
          lat=as.numeric(as.character(twitData[3,]))
          long =as.numeric(as.character(twitData[2,]))
#          print (lat);
#          print (long);
#          long <- runif(1, -120, -80)
#          lat <- runif(1, 30, 45)
          coorD = data.frame(lat=lat, long=long)
          
        ## Create a vector of colors, counties for which we don't have any data will
        ## be colored grey, others blue or red depending on who "won" that county.
        ## nclr should be 1 minus the number of fixedbreaks in the classintervals function */
        states <- map('state',plot=F)
        
        plotvar <- stateMetrics$Size
        nclr <- 8
        plotclr <- brewer.pal(nclr,"RdBu")
        class <- classIntervals(plotvar, nclr, style="fixed",fixedBreaks=c(0,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1))
        
        colcode <- findColours(class, plotclr)
        col <- rep("grey",length(states$names))
        col[stateMetrics$order] <- colcode
        
        map(database = "state",col ="white", fill = TRUE, bg="darkgray", mar = c(4.1, 4.1, par("mar")[3], 0.1))
        m <- map('state',fill=T,plot=F)
        polygon(m$x,m$y,col=colcode,border=NA)
        map('state',col="darkgrey",add=T)
        
        score = as.numeric(as.character(twitData[4,]))
        if(score<0.5){
          colorss="blue"
        }else{
          colorss="red"
        }
        points(coorD$long, coorD$lat, col = "black", cex = 1, pch = 21, bg = colorss)
      
        legend(-78, 33, c("(0, 0.125)","[0.125, 0.25)","[0.25, 0.5)","[0.5, 0.625)","[0.625, 0.75)","[0.75, 0.875)","[0.875, 1)"),
               fill=attr(colcode, "palette"), border="black", cex=0.75, bty="o")
      
        title(main=twitData[1,], sub="",
              cex.main = 0.75,font.main= 3, col.main= "Black",
              cex.sub = 2, font.sub = 3, col.sub = "Black")
      
      testit(3)
}
    }
  }
}
}