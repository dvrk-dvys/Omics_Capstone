setwd("C:/Graham/J_and_J/Lung Cancer/TCGA/LUSC/Public TCGA New/GDCdata/TGCA-LUSC-Subtype work/Basal Versus Normal")

transpose1<- read.csv("merged LUSC corrected cleaned Basal vs Normal.csv", colClasses = "character")

sucess1<-t(transpose1)

write.table(sucess1, file= "merged LUSC corrected cleaned Basal vs Normal transp.txt", quote= FALSE, col.names=FALSE, sep = "," )
