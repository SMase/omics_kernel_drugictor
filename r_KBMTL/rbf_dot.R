# Created by: shogo_mase
# Created on: 2018/06/22

kernel <- function(train1, train2, gamma) {
    sigma <- gamma
    exp(-sum((train1 - train2)^2 / (2 * sigma^2)))
}