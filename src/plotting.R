library(ggplot2)

nn <- read.csv('2022-07-07_19-28-46_nn.csv', header = F)
random <- read.csv('2022-07-07_19-28-46_random.csv', header = F)
ts <- read.csv('2022-07-07_19-28-46_ts.csv', header = F)
blr <- read.csv('2022-07-07_20-24-34_blr.csv', header = F)

nn$Model <- "Neural Network"
nn$Step <- 1:nrow(nn)
names(nn)[names(nn) == 'V1'] <- 'TotalRegret'
random$Model <- "Random"
random$Step <- 1:nrow(random)
names(random)[names(random) == 'V1'] <- 'TotalRegret'
ts$Model <- "Thompson Sampling"
ts$Step <- 1:nrow(ts)
names(ts)[names(ts) == 'V1'] <- 'TotalRegret'
blr$Model <- "Bayesian Linear Regression"
blr$Step <- 1:nrow(blr)
names(blr)[names(blr) == 'V1'] <- 'TotalRegret'

graph_data <- rbind(nn, random, ts, blr)
names(graph_data)[names(graph_data) == 'V1'] <- 'TotalRegret'

# Density Plot
ggplot(data=graph_data, aes(x=TotalRegret, color=Model)) +
  geom_density(alpha=.3) + theme(legend.position = 'bottom')
ggsave('Density Plots.png')

# Over time smoothed Line
ggplot(data=graph_data, aes(x=Step, y=TotalRegret, group=Model, color=Model)) +
  geom_smooth()
ggsave('all smoothed.png')
ggplot(data=nn, aes(x=Step, y=TotalRegret)) +
  geom_smooth() + labs(title = "Neural Network Model Regret Over Time")
ggsave('nn regret over time.png')
ggplot(data=random, aes(x=Step, y=TotalRegret)) +
  geom_smooth() + labs(title = "Random Model Regret Over Time")
ggsave('random regret over time.png')
ggplot(data=ts, aes(x=Step, y=TotalRegret)) +
  geom_smooth() + labs(title = "Thompson Sampling Model Regret Over Time")
ggsave('ts regret over time.png')
ggplot(data=blr, aes(x=Step, y=TotalRegret)) +
  geom_smooth() + labs(title = "Bayesian Linear Regression Model Regret Over Time")
ggsave('blr regret over time.png')

# T-tests two-tailed, everything
regrets <- list(nn$TotalRegret, blr$TotalRegret, random$TotalRegret, ts$TotalRegret)
regret_names <- c('nn', 'blr', 'random', 'ts')
combos <- combn(seq(length(regrets)), 2)
for (i in seq(ncol(combos))) {
  print(paste0('T-test between: ', regret_names[combos[,i][1]], ' and ', regret_names[combos[,i][2]]))
  t <- t.test(regrets[[combos[,i][1]]], regrets[[combos[,i][2]]])
  print(t)
}
# T-tests one tailed, less than random
t.test(ts$TotalRegret, random$TotalRegret, alternative = 'less')
t.test(blr$TotalRegret, random$TotalRegret, alternative = 'less')
t.test(nn$TotalRegret, random$TotalRegret, alternative = 'less')

