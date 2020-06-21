# Draws points from a csv file and ellipse from a Json file
# DISCLAIMER: Currently only supports 2D data
library(ggplot2)
library(ggforce)
library(optparse)
library(tidyr)
library(rjson)


gmm2Ellipses <- function(json_path)
{
    G <- fromJSON(file=json_path)[["gaussians"]]
    ellipses <- NULL
    for (g_idx in 1:length(G))
    {
                                        # Mean
        mu <- G[[g_idx]][["mu"]][["values"]]
        if (length(mu) != 2)
        {
            stop("Gaussian is not 2 dimensional")
        }
        x0 <- mu[1]
        y0 <- mu[2]
                                        # Covariance
        covar <- G[[g_idx]][["covar"]][["values"]]
        A <- covar[[1]][1]
        B <- covar[[1]][2]
        C <- covar[[2]][2]
                                        # from: https://cookierobotics.com/007/
        sqrtdisc <- sqrt(((A-C)/2)**2+B**2)
        l1 <- (A+C)/2 + sqrtdisc
        l2 <- (A+C)/2 - sqrtdisc
        angle <- 0
        if (B == 0 && A < C) {
            angle <- pi/2
        }
        else if (B != 0) {
            angle <- atan2(l1-A,B)
        }
                                        # 0.95 Confidence ellipse
        a <- sqrt(5.991 * l1)
        b <- sqrt(5.991 * l2)
                                        # Ellipse data
        tmp_df <- data.frame(idx=g_idx, x0=x0, y0=y0, a=a, b=b, angle=angle)
        if (is.null(ellipses)) {
            ellipses <- tmp_df
        }
        else {
            ellipses <- rbind(ellipses, tmp_df)
        }
    }
    ellipses$idx <- as.factor(ellipses$idx)
    ellipses
}


# TODO add attribution of labels (based on GMM)
# TODO think about higher dimensions
# TODO add expected gaussians
option_list <- list(
    make_option(c("-s","--samples"), type="character", default=NA,
                help="A csv file containing points drawn from the distribution to plot"),
    make_option(c("-o","--output"), type="character", default="gmm.png",
                help="The path for the output image")
    )
parser <- OptionParser(usage="%prog [options] <ellipses.json>",
                       option_list = option_list)
cmd <- parse_args(parser, commandArgs(TRUE), positional_arguments=1)

ellipses <- gmm2Ellipses(cmd$args[1])

points <- NULL
if (!is.na(cmd$options$samples))
{
    points <- read.csv(cmd$options$samples)
    if (ncol(points) != 2)
    {
        stop("Only support 2 dimensional data")
    }
}

g <- ggplot()
g <- g + geom_ellipse(data=ellipses, mapping=aes(x0=x0,y0=y0,a=a,b=b,color=idx,angle=angle,fill=idx),alpha=0.3)
if (!is.null(points))
{
    g <- g + geom_point(data =points, mapping= aes(x=col1,y=col2), size=0.5)
}
## g <- g + facet_grid(score_type ~ ., scales = "free")
ggsave(cmd$options$output)
