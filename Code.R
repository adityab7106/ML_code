
############ Feature Hashing Vectorization ############

library(text2vec)
library(data.table)
library(magrittr)

prep_fun = tolower
tok_fun = word_tokenizer

it_train = itoken(data_train$text, 
                  preprocessor = prep_fun, 
                  tokenizer = tok_fun, 
                  #ids = data_train$SN_Mod, 
                  progressbar = FALSE)

vocab = create_vocabulary(it_train)

vectorizer = vocab_vectorizer(vocab)
dtm_train = create_dtm(it_train, vectorizer)

it_test = tok_fun(prep_fun(data_test$text))

it_test = itoken(it_test, progressbar = FALSE)


dtm_test = create_dtm(it_test, vectorizer)


stop_words = c("i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours")

vocab = create_vocabulary(it_train, stopwords = stop_words)

pruned_vocab = prune_vocabulary(vocab, 
                                term_count_min = 10, 
                                doc_proportion_max = 0.5,
                                doc_proportion_min = 0.001)
vectorizer = vocab_vectorizer(pruned_vocab)
# create dtm_train with new pruned vocabulary vectorizer
dtm_train  = create_dtm(it_train, vectorizer)

dtm_test = create_dtm(it_test, vectorizer)



#### N-grams of 2-grams ###### 
vocab = create_vocabulary(it_train, ngram = c(1L, 2L))
vocab = prune_vocabulary(vocab, term_count_min = 10, 
                         doc_proportion_max = 0.5)

bigram_vectorizer = vocab_vectorizer(vocab)

dtm_train = create_dtm(it_train, bigram_vectorizer)
dtm_test = create_dtm(it_test, bigram_vectorizer)

#### Trying some Feature Hashing ##########
library(glmnet)
h_vectorizer = hash_vectorizer(hash_size = 2 ^ 14, ngram = c(1L, 2L))

dtm_train = create_dtm(it_train, h_vectorizer)
dtm_test = create_dtm(it_test, h_vectorizer)

glmnet_classifier = cv.glmnet(x = dtm_train, y = data_train[['real_fake']], 
                              family = 'binomial', 
                              alpha = 1,
                              type.measure = "auc",
                              nfolds = 5,
                              thresh = 1e-3,
                              maxit = 1e3)


predictions <- predict(glmnet_classifier, newx = dtm_test, s = "lambda.min", type = "response")
predicted_classes <- ifelse(predictions > 0.5, 1, 0)  

# Calculate accuracy
accuracy <- mean(predicted_classes == data_test$real_fake)
print(paste("Accuracy:", round(accuracy * 100, 2), "%"))




############ Random Forest Code #############
library(randomForest)

bag.AP=randomForest(as.factor(real_fake) ~ Characters+I(Characters^2)+total_words+I(total_words^2)+total_sentences+I(total_sentences^2)+num_bad_words+I(num_bad_words^2)+pos_neg+ave_sentiment+num_adj+syllables+I(pos_neg^2)+I(ave_sentiment^2)+I(num_adj^2)+I(syllables^2),data=data_train,mtry=3,importance=TRUE)

yhat.bag = predict(bag.AP,newdata=data_test)
table(yhat.bag, data_test$real_fake)

mean(yhat.bag == data_test$real_fake)








