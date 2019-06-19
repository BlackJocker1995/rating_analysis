#coding:utf-8
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import auc
import matplotlib.pyplot as plt

def read_csv(path=r'./ml-latest-small/'):
    """
    读取csv文件
    :param path:文件路径
    :return:返回数据
    """
    try:
        data = pd.read_csv(path+'ratings.csv')
    except:
        print('Open file error')
    return data

def compute_f_mat(mat_rat,user_count,movie_count):
    """
    计算f矩阵
    :param mat_rat: 经过得分筛选后的0-1矩阵
    :param user_count: 用户打分统计表
    :param movie_count: 电影被打分统计表
    :return: 评分矩阵f
    """
    temp = np.zeros(mat_rat.shape)
    temp2 = np.zeros(mat_rat.shape)
    for i in range(0, movie_count.shape[0], 1):
        temp[:, i] = mat_rat[:, i] / user_count
    for j in range(0, user_count.shape[0], 1):
        temp2[j, :] = temp[j, :] / movie_count
    D = np.dot(mat_rat.T, temp2)

    f = np.dot(D, mat_rat.T).T

    return f

def assessment(test,f,movie_index,user_count):
    """
    使用测试集进行评估r
    :param test:测试机
    :param f:评估矩阵
    :param movie_index:电影索引
    :param user_count:用户评分统计
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
            r = index / (movie_index.shape[0] - user_count[row.userId - 1])
            all_grop.append(r)
            print(r)
        except:
            print('error')

    print('average r: {0}'.format(np.array(all_grop).mean()))

def roc_pic(f_mat,user_count,mat_rat,mat_dislike,num = 50):
    """
    绘制ROC曲线
    :param f_mat:f评分矩阵
    :param user_count:用户评分表
    :param mat_rat: 经过得分筛选后的0-1矩阵
    :param mat_dislike: 用户不喜欢矩阵
    :param num: 迭代次数
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
            recommond_movie = sort_result[user,0:recommond_num]#推荐电影
            user_like = np.where(mat_rat[user,:] == 1)[0]
            user_dislike = np.where(mat_dislike[user,:] == 1)[0]

            like = np.intersect1d(recommond_movie, user_like)
            dis_like = np.intersect1d(recommond_movie, user_dislike)
            if len(user_dislike) ==0:
                fprs[user] = 0 #存在有人没有不喜欢的电影的情况
            else:
                fprs[user] = len(dis_like) / len(user_dislike)
            tprs[user] = len(like) / len(user_like)
            # print('like: {0} | {1}'.format(len(like),len(user_like)))

        th_fprs[i] = fprs.mean()
        th_tprs[i] = tprs.mean()
        #print('once fpr: {0}   tpr: {1}'.format(th_fprs[i] ,th_tprs[i]))
    roc_auc = auc(th_fprs,th_tprs) #计算auc的值
    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(th_fprs, th_tprs, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
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

    userId_col = data['userId']#取出userid
    movieId_col = data['movieId']#取出movieid

    user_count = np.array(userId_col.value_counts())#数目统计，每个元素代表该位置的个数
    movie_count = np.array(movieId_col.value_counts())#数目统计，每个元素代表该位置的个数
    movie_index = np.array(movieId_col.value_counts().index)

    userId_max = user_count.shape[0]#总数目
    movieId_max = movie_count.shape[0]#总数目


    mat = np.zeros([userId_max, movieId_max])#生成空矩阵

    #对用户的评分进行统计
    for row in train.itertuples(index=True, name='Pandas'):
        mat[row.userId - 1, np.where(movie_index == row.movieId)[0][0]] = row.rating
    #将评分小于threshold的归零
    mat_like = (mat > threshold) + 0
    mat_dislike = ((mat > 0) + 0) * ((mat <= threshold)+0)
    f_mat = compute_f_mat(mat_like,user_count,movie_count)

    assessment(test,f_mat,movie_index,user_count)

    roc_pic(f_mat,user_count,mat_like,mat_dislike)

if __name__ == '__main__':
    data = read_csv()
    analysis(data)
