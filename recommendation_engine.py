import warnings
import pandas as pd
import numpy as np
import re
import string
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise import SVD, Reader, Dataset
from surprise import accuracy
from sklearn import preprocessing

warnings.filterwarnings("ignore")

stop_words = set(stopwords.words('english'))

def preprocess(df, colname):
    """
    This function is used to preprocess the action plan names in the dataframe it does
    converting the action plan name to lower case,keeping only the alphabetical characters,removing punctuations
    and white spaces , lematizing and tokenzing the text
    :param df: Dataframe with Action plan details
    :param colname: Column name of the Action plan title in the Dataframe
    :return: returns a new dataframe with same as original dataframe but
    having a additional column which consists of cleaned text from original action plan name
    """
    new_df = df
    col_names_original = colname + "_orginal"
    new_df[col_names_original] = new_df[colname]
    for plan in new_df[colname]:
        action_plan = plan.lower()
        action_plan = re.sub("[^a-zA-Z+]", " ", action_plan)
        action_plan = "".join([ap for ap in action_plan if ap not in string.punctuation])
        action_plan = action_plan.replace("  ", " ")
        action_plan = word_tokenize(action_plan)
        action_plan = [i for i in action_plan if not i in stop_words]
        lemma = WordNetLemmatizer()
        action_plan = [lemma.lemmatize(lem_word) for lem_word in action_plan]
        action_plan = " ".join(action_plan)
        new_df[colname] = new_df[colname].replace(plan, action_plan)
    return new_df


def tfidf_preprocess(df, colname):
    """
    This function is used to vectorize the preprocessed text and calculate Cosine Similarities
    :param df: Data frame of the action plans text which were cleaned
    :param colname: Column name of the action plan
    :return: Consine similarity and Indices
    """
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df[colname])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(df.index, index=df[colname]).drop_duplicates()
    return cosine_sim, indices


def content_recommender(orginal_title, df, cosine_sim, indices, colname, sim_score_limiter=25):
    """
    This function is used to take the cosine similarities of the action plans and sort them and
    retrive the best 25 results by default

    :param orginal_title: Title of the action plan that you are searching for
    :param df: Data frame which have Action Plan details
    :param cosine_sim: Cosine similarity scores
    :param indices: consine similarity indexes
    :param colname: Name of the Column consists of Action Plams
    :param sim_score_limiter: Number of Action_plans to be subsetted for next stage of collobarative filitering ,
    by default this parameter is set to 25
    :return: The slice of a dataframe with Action Plan titles
    """
    col_name_original = colname + "_orginal"
    title = df.loc[df[col_name_original] == orginal_title, colname].iloc[0]
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[0:sim_score_limiter + 1]
    ap_indices = [i[0] for i in sim_scores]
    return df[col_name_original].iloc[ap_indices]


def get_content_recommendations(df, colname, ap_name):
    """
    This function retrives the content based recommendations from the Action Plans

    :param df: Data frame consists of Action plan details
    :param colname: Name of the column that has Action Plan titles
    :param ap_name: Name of Action plan that we want to retrieve recommendations for
    :return:
    """
    ap_details_processed = preprocess(df, colname)
    cosine_sim_scores, indices = tfidf_preprocess(ap_details_processed, colname)
    df_recommendations = content_recommender(ap_name, ap_details_processed, cosine_sim_scores, indices, colname)
    return df_recommendations


def train_collaborative_model(scaler, df_ratings):
    """
    This function retrives the Collaborative based recommendations from the ratings matrix
    :param scaler: Scale in which ratings are given For example ratings were given in a scale 0-10 then scaler is 10
    :param df_ratings: Triplets of user id,Action Plan Id,Rating
    :return: Trained SVD Model
    """
    reader = Reader(rating_scale=(0, scaler))
    data = Dataset.load_from_df(df_ratings, reader)
    svd = SVD()
    trainset = data.build_full_trainset()
    svd.fit(trainset)
    testset = trainset.build_anti_testset()
    predictions = svd.test(testset)
    print(accuracy.rmse(predictions, verbose=True))
    return svd


def get_hybrid_recommendations(svd_model, df_ap_details, colname, ap_name, user_id, recommendations_limiter):
    """
    This function retrives the recommendations based on both title of the action plan and the user's history for the ratings

    :param svd_model:  Trained Single Value decomposition model
    :param df_ap_details: Dataframe that consists of Action plan details
    :param colname: Name of the column in the data frame that have titles of the action plans
    :param ap_name: Name of the Action Plan
    :param user_id: User Id
    :param recommendations_limiter: Number of recommendations
    :return: Recommendation list
    """
    apps_index = get_content_recommendations(df_ap_details, colname, ap_name)
    app = df_ap_details.iloc[apps_index.index]
    app['est'] = [svd_model.predict(user_id, x).est for x in app['ap_id']]
    min_max_scaler = preprocessing.MinMaxScaler()
    app['match'] = min_max_scaler.fit_transform(app[['est']])
    app = app.sort_values("est", ascending=False)
    return app.head(recommendations_limiter)

