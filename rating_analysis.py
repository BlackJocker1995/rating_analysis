#coding:utf-8
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import auc
import matplotlib.pyplot as plt

def read_csv(path=r'./ml-latest-small/'):
    """
    read csv file
    :param path:file path
    :return:csv data as DataFrame
    """
    try:
        data = pd.read_csv(path+'ratings.csv')
    except:
        print('Open file error')
    return data

def compute_f_mat(mat_rat,user_count,movie_count):
    """
    compute the f matrix
    :param mat_rat: user`s rating matrix([user number,movie number]) where 1 means user likes the index movie.
    :param user_count: statistics of moive numbers that user have watch.
    :param movie_count: statistics of user numbers that movie have been rated.
    :return: f matrix
    """
    temp = (mat_rat / user_count.reshape([-1,1]) )/ movie_count.reshape([1,-1])
    D = np.dot(mat_rat.T, temp)

    f = np.dot(D, mat_rat.T).T

    return f

def assessment(test,f,movie_index,user_count):
    """
    compute assemssment r using test data
    :param test:test data(dataFrame)
    :param f:f matrix
    :param movie_index:index of movie
    :param user_count:statistics of moive numbers that user have watch.
    :return:None
    """
    sort_result = np.argsort(-f, axis=1)

    all_grop = []
    for row in test.itertuples(index=True, name='Pandas'):
        if row.rating < 3:
            print('rating is smaller that threshold')
            continue
        score = sort_result[row.userId - 1, :]
        try:
            index = np.where(score == np.where(row.movieId == movie_index)[0][0])[0][0]
            r = index / (movie_index.shape[0])
            all_grop.append(r)
            print(r)
        except:
            print('error')
    plt.hist(np.array(all_grop),bins=100,facecolor='black',edgecolor='black',alpha=1,histtype='bar')
    plt.show()
    print('average r: {0}'.format(np.array(all_grop).mean()))

def roc_pic(f_mat,user_count,mat_rat,mat_dislike,num = 50):
    """
    drawn roc figure
    :param f_mat:f matrix
    :param user_count:statistics of moive numbers that user have watch.
    :param mat_rat: user`s rating matrix([user number,movie number]) where 1 means user likes the index movie.
    :param mat_dislike: user`s dislike matrix([user number,movie number]) where 1 means user likes the index movie.
    :param num: number of looping
    :return:None
    """
    threshold_rate = np.linspace(0,1,num)

    sort_result = np.argsort(-f_mat, axis=1)
    th_fprs = np.zeros(num)
    th_tprs = np.zeros(num)
    for i,threshold in enumerate(threshold_rate):
        recommond_num = int(mat_rat.shape[1] * threshold)
        fprs = np.zeros(user_count.shape[0])
        tprs = np.zeros(user_count.shape[0])
        for user in range(user_count.shape[0]):
            recommond_movie = sort_result[user,0:recommond_num]#recommand movies
            user_like = np.where(mat_rat[user,:] == 1)[0]
            user_dislike = np.where(mat_dislike[user,:] == 1)[0]

            like = np.intersect1d(recommond_movie, user_like)
            dis_like = np.intersect1d(recommond_movie, user_dislike)
            if len(user_dislike) ==0:
                fprs[user] = 0 #There are some users do not have unfavoraable movie
            else:
                fprs[user] = len(dis_like) / len(user_dislike)
            tprs[user] = len(like) / len(user_like)

        th_fprs[i] = fprs.mean()
        th_tprs[i] = tprs.mean()
        #print('once fpr: {0}   tpr: {1}'.format(th_fprs[i] ,th_tprs[i]))
    roc_auc = auc(th_fprs,th_tprs) #compute the roc value
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(th_fprs, th_tprs, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

def analysis(data,threshold = 3):
    train, test, _, _ = train_test_split(data, data['userId'], test_size=0.1)

    userId_col = data['userId']#get userid
    movieId_col = data['movieId']#get movieid

    user_count = np.array(userId_col.value_counts())#count number，every element of array meas number of this ID index
    movie_count = np.array(movieId_col.value_counts())#count number，every element of array meas number of this ID index
    movie_index = np.array(movieId_col.value_counts().index)

    userId_max = user_count.shape[0]#all number
    movieId_max = movie_count.shape[0]#all number


    mat = np.zeros([userId_max, movieId_max])#create empty matrix

    #count the rating of users
    for row in train.itertuples(index=True, name='Pandas'):
        mat[row.userId - 1, np.where(movie_index == row.movieId)[0][0]] = row.rating
    #set zero when elements smaller that threshold
    mat_like = (mat > threshold) + 0
    mat_dislike = ((mat > 0) + 0) * ((mat <= threshold)+0)
    f_mat = compute_f_mat(mat_like,user_count,movie_count)

    assessment(test,f_mat,movie_index,user_count)

    roc_pic(f_mat,user_count,mat_like,mat_dislike)

if __name__ == '__main__':
    data = read_csv()
    analysis(data)
