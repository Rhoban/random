library(ggplot2)
library(optparse)
library(tidyr)

option_list <- list(
    make_option(c("-o","--output"), type="character", default="scores.png",
                help="The path for the output image")
    )
parser <- OptionParser(usage="%prog [options] <score_logs.csv>",
                       option_list = option_list)
cmd <- parse_args(parser, commandArgs(TRUE), positional_arguments=1)

data <- read.csv(cmd$args[1])
data <- gather(data, key=score_type, value=score, logLikelihood:BIC, factor_key = TRUE)

g <- ggplot(data, aes(x=nbClusters,y=score,group=covMatType, fill=covMatType, color= covMatType))
g <- g + geom_point()
g <- g + facet_grid(score_type ~ ., scales = "free")
ggsave(cmd$options$output)
