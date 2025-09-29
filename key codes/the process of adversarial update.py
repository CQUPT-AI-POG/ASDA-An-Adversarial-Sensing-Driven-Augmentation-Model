# After getting the features, input them manually, and then get the representation of each tweet corresponding to the features.
from tool_llm_class import LLM
from tool_read_data import tool_read_data_ori_train_sms, tool_read_data_all_sms, write_data_feature_vector
from tool_ASDA import Tweet, ASDA
import random

# This is the update process for the recorded latent features of SMS.

# Threshold is 0.125, accuracy is 0.9591836734693877, recall is 0.7083333333333334
sms_feature_interaction_1 = [
    "Spam exhibits chaotic font types: using multiple different fonts and sizes in the same email, creating visual noise.",  # Accuracy: 0, Identification rate: 0.0
    "Spam exhibits irregular spacing or symbol insertion: for example, 'F R E E' or 'W_I_N' to evade keyword detection.",  # Accuracy: 0, Identification rate: 0.0
    "Spam contains a large amount of commercial promotion and advertising words.",  # Accuracy: 1.0, Identification rate: 0.4897959183673469
    "Spam exhibits an excessive pile-up of marketing terms, such as 'super value', 'exclusive', 'officially certified' appearing repeatedly.",  # Accuracy: 0, Identification rate: 0.0
    "Spam exhibits hidden elements in HTML emails: hidden hyperlinks, transparent images, used to track recipient behavior.",  # Accuracy: 0, Identification rate: 0.0
    "Spam often features urgent language.",  # Accuracy: 0.8947368421052632, Identification rate: 0.3877551020408163
    "Spam exhibits abnormal line spacing and paragraphs: large blocks of white space or excessively long continuous text.",  # Accuracy: 0, Identification rate: 0.0
    "Spam is excessively pictorial: the body is almost all images with very little text (to prevent text filter identification).",  # Accuracy: 0, Identification rate: 0.0
]

sms_feature_interaction_2 = [
    "Spam contains a large amount of commercial promotion and advertising words.",
    "Spam often features urgent language.",
]
# Accuracy 0.875, recall is 1.0

# Threshold is 0.08333333333333333, accuracy is 0.9795918367346939, recall is 0.875
sms_feature_interaction_3 = [
    "Spam contains a large amount of commercial promotion and advertising words.",  # Accuracy: 1.0, Identification rate: 0.40816326530612246
    "Spam exaggerates benefits or effects.",  # Accuracy: 1.0, Identification rate: 0.16326530612244897
    "Spam often features urgent language.",  # Accuracy: 1.0, Identification rate: 0.24489795918367346
    "Spam contains a large number of external links, sometimes using shortened URLs or strange domain names.",  # Accuracy: 1.0, Identification rate: 0.14285714285714285
    "Spam contains high-risk attachments such as executable files (.exe), macro files (.docm), etc.",  # Accuracy: 0, Identification rate: 0.0
    "Spam titles or content are all in uppercase to enhance visual impact.",  # Accuracy: 1.0, Identification rate: 0.04081632653061224
    "Spam mixes in special characters or replaces letters (e.g., 'V1agra' for 'Viagra') to bypass filters.",  # Accuracy: 0, Identification rate: 0.0
    "Spam contains sensitive words, such as 'prize', 'loan', 'promotion', etc.",  # Accuracy: 1.0, Identification rate: 0.2857142857142857
    "Spam has suspicious domain names, with spellings similar to legitimate domains (e.g., amaz0n.com).",  # Accuracy: 0, Identification rate: 0.0
    "Spam guides users to click on links and enter sensitive information (such as bank account numbers, passwords, verification codes).",  # Accuracy: 0, Identification rate: 0.0
    "Spam contains excessive advertising-related content.",  # Accuracy: 1.0, Identification rate: 0.3469387755102041
    "Spam exhibits unusual language mixing, incorporating multiple languages or using unnatural translated grammar, usually from machine translation.",  # Accuracy: 0, Identification rate: 0.0
]

sms_feature_interaction_4 = [
    "Spam contains a large amount of commercial promotion and advertising words.",
    "Spam exaggerates benefits or effects.",
    "Spam often features urgent language.",
    "Spam contains a large number of external links, sometimes using shortened URLs or strange domain names.",
    "Spam titles or content are all in uppercase to enhance visual impact.",
    "Spam contains sensitive words, such as 'prize', 'loan', 'promotion', etc.",
    "Spam contains excessive advertising-related content.",
]
# Accuracy is 0.9795918367346939, recall is 0.9583333333333334


def update_Feature():
    # 1. Read data
    train_data_list, train_label_list = tool_read_data_ori_train_sms()
    # Store the original data into Tweet objects
    Tweet_list = []
    for pos in range(len(train_data_list)):
        tweet = Tweet()
        tweet.content = train_data_list[pos]
        tweet.label = train_label_list[pos]
        Tweet_list.append(tweet)

    random.shuffle(Tweet_list)

    asda = ASDA()
    asda.get_Data(Tweet_list)
    asda.get_Feature(sms_feature_interaction_2)
    llm = LLM()
    asda.data_text_class = llm.init_get_attribute_ssm(data=asda.data_text_class, feature=sms_feature_interaction_2)

    # Not looking at this
    before_right, before_recall, after_right, after_recall = asda.evaluate_Attributes()
    # print(f"Accuracy and recall before and after removal are {before_right, before_recall, after_right, after_recall}")


# Tag all data with feature vectors
def get_Feature_Vector_Alldata():
    int_data = 1200
    train_data_list, train_label_list = tool_read_data_all_sms()
    Tweet_list = []
    for pos in range(len(train_data_list)):
        tweet = Tweet()
        tweet.content = train_data_list[pos]
        tweet.label = train_label_list[pos]
        Tweet_list.append(tweet)
    random.seed(2025)
    random.shuffle(Tweet_list)
    Tweet_list = Tweet_list[int_data - 30:int_data]
    asda = ASDA()
    asda.get_Data(Tweet_list)
    asda.get_Feature(sms_feature_interaction_4)
    llm = LLM()
    asda.data_text_class = llm.init_get_attribute_ssm(data=asda.data_text_class, feature=sms_feature_interaction_4)
    write_data_feature_vector(asda.data_text_class, int_data)


if __name__ == '__main__':
    # update_Feature()
    get_Feature_Vector_Alldata()