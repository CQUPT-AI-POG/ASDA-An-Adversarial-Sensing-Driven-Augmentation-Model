class Tweet:
    def __init__(self):
        self.content = None
        self.label_int = None
        self.label = None
        self.attribute = None
        self.anomalyValue = None

class Feature:
    def __init__(self):
        self.content = None

class ASDA:
    def __init__(self):
        self.data_text_class = None

    def get_Data(self, data):
        self.data_text_class = data

    def get_Feature(self, feature_list):
        self.tree_attribute = []
        for item in feature_list:
            feature = Feature()
            feature.content = item
            self.tree_attribute.append(feature)

    def evaluate_Attributes(self):
        # 1. First, determine the anomaly value for each tweet
        for tweet in self.data_text_class:
            # print(f"Tweet's attribute {tweet.attribute}")
            count_true = sum(1 for item in tweet.attribute if item == True or item == 'True')
            tweet.anomalyValue = count_true / len(tweet.attribute)
            # print(f"The anomaly value of {tweet.text} is {tweet.anomalyValue}")
            if tweet.label == 'ham':
                tweet.label_int = 0  # Represented in numeric form
            if tweet.label == 'spam':
                tweet.label_int = 1
            # print(f"Tweet's anomalyValue {tweet.anomalyValue}, actual label {tweet.label_int}")

        # 2. Determine the best threshold
        labels = [item.label_int for item in self.data_text_class]
        scores = [item.anomalyValue for item in self.data_text_class]
        # Get all unique sorted thresholds
        unique_scores = sorted(set(scores))
        best_threshold = None
        best_accuracy = -1
        best_recall = -1
        recall_count = 0
        anomaly_count = 0
        # Determine the best threshold
        for threshold in unique_scores:
            # Predict based on the threshold
            predictions = [1 if score >= threshold else 0 for score in scores]  # Higher is anomalous
            # Calculate accuracy
            correct = sum(1 for pred, true in zip(predictions, labels) if pred == true)
            accuracy = correct / len(labels)
            # print(f"Accuracy for threshold {threshold} is {accuracy}")
            # Update the best threshold
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
        # Determine the recall rate
        for tweet in self.data_text_class:
            output = tweet.attribute
            pred = 1 if tweet.anomalyValue > best_threshold else 0
            if pred == 1 and tweet.label_int == 1:
                recall_count += 1
            if tweet.label_int == 1:
                anomaly_count += 1
        best_recall = recall_count / anomaly_count  # v1 algorithm, v2 is to calculate F1-SCORE
        print(f"The most suitable threshold is {best_threshold}, accuracy is {best_accuracy}, recall is {best_recall}")

        # 2-1. Determine the best threshold, f1-score
        # labels = [item.label_int for item in self.data_text_class]
        # scores = [item.anomalyValue for item in self.data_text_class]
        # # Get all unique sorted thresholds
        # unique_scores = sorted(set(scores))
        # best_threshold = None
        # best_f1 = -1
        # best_precision = 0
        # best_recall = 0
        #
        # # Determine the best threshold
        # for threshold in unique_scores:
        # # Predict based on the threshold
        #     predictions = [1 if score >= threshold else 0 for score in scores] # Higher is anomalous
        #     tp = sum(1 for pred, true in zip(predictions, labels) if pred == 1 and true == 1)
        #     fp = sum(1 for pred, true in zip(predictions, labels) if pred == 1 and true == 0)
        #     fn = sum(1 for pred, true in zip(predictions, labels) if pred == 0 and true == 1)
        #
        #     precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        #     recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        #     f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        #
        #     if f1 > best_f1:
        #         best_f1 = f1
        #         best_threshold = threshold
        #         best_precision = precision
        #         best_recall = recall
        #
        # print(f"The most suitable threshold is {best_threshold}, F1-score is {best_f1}, Precision is {best_precision}, Recall is {best_recall}")

        # 3. Determine the utility of each label (the independent accuracy of each label) and remove inefficient attributes
        index_attribute = 0
        self.tree_attribute_update = []  # First, empty the updated attribute tree
        for attribute in self.tree_attribute:  # Iterate through all attributes, each attribute is a dictionary
            right_count = 0  # Count the number of recognitions as True and the data is troll
            wrong_count = 0  # Count the number of recognitions as True and the data is normal
            total_count = 0  # The total number of recognitions
            # Iterate through the output of each attribute
            for tweet in self.data_text_class:
                output = tweet.attribute  # [True, False, False...] obtained from the large model
                for index_output in range(len(output)):
                    if output[index_output] == 'True' and index_output == index_attribute:  # That is, this attribute judges an anomaly
                        if tweet.label_int == 1:  # Successfully identified anomalous tweet
                            right_count += 1
                        if tweet.label_int == 0:  # Misidentified anomalous tweet
                            wrong_count += 1
                        total_count += 1  # The total number of tweets identified
            right_rate = right_count / total_count if total_count != 0 else 0
            identify_rate = total_count / len(self.data_text_class)
            print(f"Attribute: {attribute.content}, Identification accuracy: {right_rate}, Identification rate: {identify_rate}")
            if right_rate > 0.5:  # Or compare average performance, no big difference
                self.tree_attribute_update.append(attribute)
            index_attribute += 1
        for item in self.tree_attribute_update:
            print(f"Attribute after removal is {item.content}")

        # 4. Determine the effect after updating
        indices = [i for i, x in enumerate(self.tree_attribute) if x in self.tree_attribute_update]
        # Filter the output of the tweets
        filtered_other_2d_lists = [[tweet.attribute[i] for i in indices] for tweet in self.data_text_class]
        for tweet in self.data_text_class:
            tweet.attribute = filtered_other_2d_lists[self.data_text_class.index(tweet)]
        # Recalculate accuracy and recall
        self.data_text_wrong_class = []
        wrong_data_index_list = []
        right_count = 0
        recall_count = 0
        troll_count = 0  # Used to count the total number of trolls
        for tweet in self.data_text_class:
            output = tweet.attribute
            pred = 0
            for item in output:
                if item == 'True':  # Having True means it is anomalous
                    pred = 1
            if tweet.label_int == 1:
                troll_count += 1
            if pred == 1 and tweet.label_int == 1:
                recall_count += 1
            if pred == tweet.label_int:
                right_count += 1
            if pred != tweet.label_int:
                # print(f'Incorrectly classified data: {tweet.text}')
                self.data_text_wrong_class.append(tweet)
                wrong_data_index_list.append(self.data_text_class.index(tweet))
        print(f"Updated accuracy is {right_count / len(self.data_text_class)}, recall is {recall_count / troll_count}")

        # Display incorrectly classified data
        print("Incorrectly classified data:")

        for i in range(len(self.data_text_wrong_class)):
            print(f"Data {self.data_text_wrong_class[i].content}")
            print(f"True label is: {self.data_text_wrong_class[i].label}")
            print(f"Judged attributes {self.data_text_wrong_class[i].attribute}")
        before_right, before_recall, after_right, after_recall = best_accuracy, best_recall, right_count / len(
            self.data_text_class), recall_count / len(self.data_text_class)
        # Display the index of the incorrectly classified data for calculation
        print(f"The index of the incorrect data is {wrong_data_index_list}")
        return before_right, before_recall, after_right, after_recall