install.packages('ggridges')
install.packages('readxl')
library(ggplot2) 
library(ggridges) 
library(readxl)
#theme_set(theme_ridges()) 
library(RColorBrewer) 
Colormap<- colorRampPalette(rev(brewer.pal(11,'Spectral')))(32) 


data <-read_excel("neimeng.xlsx",sheet = 1)
N <-ncol(data)-1


ggplot(data, aes(x = 'cases',y = 'month')) + 
  geom_density_ridges_gradient(aes(fill = ..x..), scale = 2, size = 0.3 ) + 
  scale_fill_gradientn(colours=Colormap,name = "Temp. [F]")
#p + scale_y_discrete(expand = expansion(mult = c(0.01, 0.25)),labels = w)


w <- c("Dev","Nov","oct","Sep","Aug","Jul","Jun","May","Apr","Mar","Feb","Jan")

ggplot(data, aes(x =cases,y = month,  fill = month)) + 
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_discrete(expand = expansion(mult = c(0.01, 0.25)),labels = w)+
  geom_density_ridges_gradient(aes(fill = ..x..), scale = 2, size = 0.3) + 
  #theme_minimal()+theme(legend.position = "none") + 
  scale_fill_gradientn(colours=Colormap,name = "Incidence")+
  xlab("Monthly incidence of human brucellosis") +
  theme_classic()+
  theme(panel.grid.major=element_line(colour=NA),
        panel.background = element_rect(fill = "transparent",colour = NA),
        plot.background = element_rect(fill = "transparent",colour = NA),
        panel.grid.minor = element_blank())







