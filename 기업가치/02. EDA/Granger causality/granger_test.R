library(lmtest)
library("dplyr")                         
######
df = read.csv(file='ACN.csv')

target_df <- df[,1:(ncol(df)-2)]

target_df$yhat<-data.table::shift(target_df, n=1, fill=NA, type=c("lead"), give.names=FALSE)[45][[1]]
target_df$yhat_re<-data.table::shift(target_df, n=1, fill=NA, type=c("lag"), give.names=FALSE)[45][[1]]
target_df$yhat_re[1] <- target_df$yhat_re[2]

new_df <- slice(target_df, 1:(n() - 1))

# 인과관계 방향 확인하는 test
g <- list()
g <- grangertest(new_df$yhat_re~new_df$target, order=1)
g

g <- list()
g <- grangertest(new_df$target~new_df$yhat, order=1)
g


out = NULL
g <- list()
for (i in colnames(new_df[,2:(ncol(new_df)-3)])) {
  g <- grangertest(new_df$yhat_re~new_df[[i]], order=1)
  #  cat(i, ":", round(g$'Pr(>F)'[2], digits=3), "\n ")
  col <- c((i))
  lag_0_reserve <- c(round(g$'Pr(>F)'[2], digits =3))
  data<- data.frame(col, lag_0_reserve)
  out <- rbind(out, data)
}
# out


out2 = NULL
g2 <- list()
for (i in colnames(new_df[,2:(ncol(new_df)-3)])) {
  g2 <- grangertest(new_df[[i]]~new_df$yhat, order=1)
  # cat(i, ":", round(g2$'Pr(>F)'[2], digits=3), "\n ")
  col <- c((i))
  lag_0 <- c(round(g2$'Pr(>F)'[2], digits =3))
  data2<- data.frame(col, lag_0)
  out2 <- rbind(out2, data2)
}
# out2

out_final <- merge(out,out2, by.x='col', by.y ='col')
write.csv(out_final, 'granger_final.csv', row.names=FALSE)
