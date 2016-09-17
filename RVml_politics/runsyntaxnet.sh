#!/bin/bash


while [[ 1 == 1 ]]; do
   if [[ -e /Users/pal004/PoliticPredictors/forsyntaxnet.json ]]; then
      sleep 1
      rm /Users/pal004/PoliticPredictors/syntaxnetoutput.formodel.txt
      cd /Users/pal004/anaconda/syntaxnet-custom/models/syntaxnet
      cat /Users/pal004/PoliticPredictors/forsyntaxnet.json  | syntaxnet/demo.sh 2>/Users/pal004/PoliticPredictors/blah2.txt | grep -v "^Input" >  /Users/pal004/PoliticPredictors/tmp.txt
      rm /Users/pal004/PoliticPredictors/forsyntaxnet.json
      mv /Users/pal004/PoliticPredictors/tmp.txt /Users/pal004/PoliticPredictors/syntaxnetoutput.formodel.txt
      sleep 1
   fi
done


