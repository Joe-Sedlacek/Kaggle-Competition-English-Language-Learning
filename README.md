# English Language Learning Vocabulary Scoring
## Overview
1. Introduction
2. Results of Initial Approach and Analysis 
3. The Secondary Approach
4. Identification of “Out of Vocabulary Words”
5. Results and Future Directions 


## 1. Introduction

Every year, many English Language Learners (ELLs) are assessed on their language skills and writing level. These frequently come in the form of standardized tests and are assessed by workers who see hundreds of essays every day. That work is tiresome, difficult, and often very subjective, as readers are forced to put numbers on typically qualitative data, such as the strength of a student’s grammar, writing comprehension, or vocabulary. When we stumbled upon a Kaggle Competition about measuring features of ELLs’ essays, we knew this would be a great area to investigate. 

Additionally, the fact that this data set revolved around assessing students in which English was a secondary language appealed to us. We all have had sufficient experience with learning new languages, such as Spanish and Danish, and understand the difficulty of learning and then writing in a new language. We also acknowledge that learning a language comes with a plethora of struggles, such as an accidental substitution of words from a more familiar language, misspelling of words that the students comprehend, and grammatical errors. The dataset we found contained a collection of essays and graded results on six features of essays, but we decided to focus on creating a model to predict specifically the vocabulary score for an essay. 

We picked to measure the vocabulary of ELLs because it is a complex underlying structure of every essay, blog post, or news article. A successful model that could measure the vocabulary strength of a piece of text would provide crucial information about the strength of any writer that you are reading. It also comes with an array of challenges that we wanted to tackle. For starters, how would you measure the strength of the vocabulary of this blog post thus far? Personally, we would measure it as average, but the writing itself is correctly performing the function it is intended to do, which is to introduce our project. Since these essays are written by language learners, we need to consider that these prompts are likely not complex prompts that require a large amount of writing and advanced terminology. Additionally, a challenge with specific language learners is addressing misspelled words or words of a different language, since they would be more common than in a typical post. Measuring the vocabulary of these students’ writing is a meaningful and important task, and we will explain in the upcoming sections how we approached this problem and the results of our work.

## 2. The Initial Approach 

When we first approached this problem, we knew the primary challenge would be selecting an appropriate approach. Many Natural Language Processing (NLP) solutions use Recurrent Neural Networks (RNNs) to process text in a sequential manner. Generally, vocabulary is not dependent on the sequence of words, but rather the types of words that are used. For ELLs writing, we expect that use of common words suggests a low level of grammar, while use of less common words suggests the opposite. Therefore, we looked for a method of classifying the complexity of the words used in each essay.

To accomplish this, we turned to the TF-IDF method. TF-IDF stands for term frequency inverse document frequency, and it measures how frequently a word is used in a document relative to its use in all documents in a pool. In our first approach, we used SKLearn’s built in TF-IDF vectorizer, which uses the following formula to create a score for each word in collection of documents:
- TF is calculated by # of times a word appears in current document
- IDF is calculated by log base e ([1 + # documents] / [1 + # documents word appears in])
- SCORE = TF * IDF

The result of this formula is that words with high frequency in the current document and low frequency overall have the highest scores, and words with low frequency in the current document and high frequency overall will have the lowest scores. Note that the score is zero for any word that does not appear in the current document. For each document, the TF-IDF vectorization produces a single vector with length equal to the total number of unique words in all documents. The vectorizer then normalizes the TF-IDF vector so that it can be compared with other vectors better. Our idea was to then use these TF-IDF vectors as the input to our neural network.

We selected this approach because we believed higher vocabulary scores would be given to writing with words that are not used as frequently. To test this theory, we used the TF-IDF vectorization of the training data, then trained a neural network with the vectorizations and the given vocabulary scores for each essay. With the data from Kaggle, we had 3,911 total essays. We did a 80/20 training/testing split on the data, so we had 3,128 essays for training and 783 essays for testing. The mean-squared-error (MSE) training loss was 0.0352. However, we needed to test the network on the testing data to get an accurate representation of its success.

We needed to create TF-IDF vectors for the testing data before we could feed it into the network. We can’t run the same SKLearn function for this, though, because TF-IDF scores depend on analyzing a collection of documents. In theory, our model needs to work for a single new input essay, so we need to reuse some information from the TF-IDF vectorization of the training data. To be precise, we use the IDF vector created in this process that indicates the frequency of each term in the training data. We then count the frequency of each of those terms in the testing data. The drawback of this approach is that any new terms in the testing data will be ignored in our process. However, there were 18,913 unique terms in the training data, so the model will still get a sizable amount of information from the testing data. Once we had term frequency (TF), we multiplied the TF vector by the precomputed IDF vector, then normalized the result. This process created full TF-IDF vectorizations for the testing data.
Once the TF-IDF vectors were computed for the testing data, we evaluated the model. The MSE loss for the testing data was 0.3054. Clearly, this is much higher than the MSE of 0.0352 which was the loss in training, so overfitting was a concern of ours. However, this result confirmed that the TF-IDF approach was viable. At this point, we then began thinking about other improvements that could be made. We didn’t want the TF-IDF vectorization to depend on a narrow selection of ELLs’ essays, so we looked for a different approach.

## 3. The Secondary Approach
	
We began to question the ways in which this model could be improved. In particular, we analyzed our primary method of categorizing every essay and the TF-IDF metric itself. TF-IDF is typically used to assess the importance of a word to a document in a corpus. In our case, we used this metric to analyze each essay in which the corpus was the entire collection of essays. However, after creating the first approach, we noticed two drawbacks with this approach. The first drawback is that the word pool was restricted to only words that were in the corpus of essays. As mentioned earlier, this would mean any new terms that were not in the training set would then be discarded in order to create the new custom TF-IDF vector to send into the network. This leads to issues, as there are some words that ELLs might not know, and could be used. Since we are aiming to measure total vocabulary skill, we need to successfully account for the use of words that aren’t typically used, such as ones that might get discarded in this first approach. 

The second drawback that we identified is that the word understanding of ELLs is likely different from the word understanding of the graders, which we are training our network to imitate. Since we are using the TF-IDF to measure the use of words that are unique to a specific essay and not for the usually intended purpose of measuring the importance to a document, we thought of a new approach that could hopefully classify words as more common or less common through the use of the most commonly used English words, by all people, and not just ELLs. 

For this new approach, we looked at a Kaggle dataset labeled “English Word Frequency” that contained the most common ⅓ million words in English on the web. For every word, it lists the frequency of the word as well. From this, we developed a new idea: creating a custom TF-IDF metric that took into consideration the number of times a word was used in a document and the frequency of the word in the English language. In order to find the new “inverse document frequency” statistic, we built it the way SKlearn. 

## 4. Identification of “Out-of-Vocabulary Words”
	
Yet another step we took to improve our initial model was how to deal with “out-of-vocabulary” words, or OOVs. An OOV is defined as a word which is not present in a training dataset or recognized by an NLP library. In our case, we used the spaCy NLP library to process OOVs. OOVs can pose as a big problem in any NLP model as they are not “natural language”, and may be mishandled by an NLP model. In the case of our problem at hand, an OOV would alter our models’ predicted score of vocabulary in a detrimental manner. The reason being is that an OOV would have a high-valued TF-IDF representation, or score, for it would appear very rarely throughout the corpus of our essays. 

For instance, let us say a student wrote the word “fhaijklal” in an essay. This word would be labeled as an OOV, for it is not an English word. Because this is not an English word, the frequency with which “fhaijkal” shows up in an English essay would be very low, resulting in a high TF-IDF score. This high TF-IDF score would result in increasing the vocabulary score for this student's vocabulary, in effect, rewarding the student for using incorrect English vocabulary. This would lead to inflating the vocabulary score for students who use incorrect English vocabulary or use non-English words. 

At first, one would think, “simply remove the OOVs”. Though this sounds like a simple fix, this would actually degrade our vocabulary predictions. This is because a misspelled word also gets flagged for being an OOV; however, we do not want to remove misspelled words because more difficult words, such as “abhorrent”, are more likely to be misspelled than common words such as “the”. If we remove a misspelled “abhorrent”, then we would not be recognizing a student’s advanced vocabulary, thus we cannot remove any OOVs out of this fear. One might then say, then simply “auto-correct” the English word like an iPhone can do. However, this would not work in our case either, for misspelled words are not the only “word” to be flagged as an OOV. In addition to misspelled words, words in other languages such as Spanish would also be flagged as OOVs as well as words that do not resemble any language (e.g. nonsense).

In order to account for all of these cases (and more) which might yield an OOV, we decided it would be best to replace the OOV with a word that has similar semantics. For example, the word “u” might be replaced with “you”, the word “bueno” might be replaced with the word “good”, and the word “rejkhaivk” in “I like to eat rejkhaivk” might be replaced with “food”. This can be seen in an example below (show example of replacing “u” and some other stuff in a short sentence…format sentence so “that” makes sense”.
	How did we do this? First, we 

Why we do it 

How we do it 

Outcome 

Considerations…

## 5. Conclusions and Future Directions
