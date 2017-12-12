Answers for conceptual questions

1a.

1. Yes, the scales of the features are important. For example the first frature ranges between 0-1 and the second feature ranges between 0-1000. Then the distance is more based on the second feature. The solution is add a weight for each feature to balance the influence of the distance.

2. Numeric feature can be calculate directly, while categorical feature can not. My thought is creating more features which are bianry. These new features are within the domain of this categorical.

3. Testing data is very important in judging the accuracy and helpful to fix our model.

4. "Supervised" means you can generally predict which feature is more important and choose a suitable model or weights for features.

5. I would pick the size of diffent part of Iris. It could be totally different between different kind of flowers. It's numeric 

1b.

1. The strengths of BOW is that this method could detect the frequency of the specific words and compare the difference. If a mail contains many bad words, it would be determined as a spam.
However, the weakness is that if a mail contains bad words but it is not a spam mail. For example, some reminders might contain bad words to ask you to avoid these. like drug, alcohol and so on.

2. The words that appear infrequent are more predictive. The words that appear more frequent are less predictive. The more frequent a word appear, the more common that it appears in a mail. So it has less weight in judging a mail.

3. I tried my classifer. The resault is always the same with KNN classifer. In my thought, the reason might be I used the most common 100 words without STOP words. I will upgrade later. In my opinion, the reason for misclassify might be various. The most common reason may be choosing the BOW. We should choose a BOW that contains most predictive words.
