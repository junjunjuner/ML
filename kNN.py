'''
kNN核心思路：选取前k个距离最近的样本中对应最多的那个标签为输入样本的标签
即遵循大多数原则
'''
from numpy import *
import operator

# 创建训练样本及对应标签
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

# inX为测试样本集，1行M列
# dataSet为训练样本集，N行M列
# labels为训练样本集中各训练样本对应的标签分类，1行N列
# k为要提取的进行比较的前k个近邻，k为奇数且小于20
def classify0(inX, dataSet, labels, k):
    # 1.计算已知类别数据集中的点与当前点的距离(平方差)
    # shape函数输出为(第一个维度个数，第二个维度个数，...)
    dataSetSize = dataSet.shape[0]     # dataSetSize表示dataSet的行数，即训练样本集的个数
    # numpy.tile(A,reps) A指待输入的数组，reps决定A重复的次数
    # tile(inX, (dataSetSize,1))表示将inx扩充为N行M列的向量（inX扩充dataSize行1列）
    # 下面四步为计算欧几里得距离
    diffMat = tile(inX, (dataSetSize,1)) - dataSet     # 对应元素相减 x1-y1,x2-y2,...
    sqDiffMat = diffMat ** 2     # 求平方 (x1-y1)^2,(x2-y2)^2,...
    # axis=1表示数据在以行为单位上进行操作
    # axis=0表示数据在以列为单位上操作
    sqDistances = sqDiffMat.sum(axis=1)   #平方差求和  (x1-y1)^2+(x2-y2)^2+...
    # 计算距离
    distance = sqDistances ** 0.5       #开平方  [(x1-y1)^2+(x2-y2)^2+...]^1/2
    # 对距离进行排序
    # argsort()函数将distance从小到大排序，并提取其对应的索引值，而后输出
    # eg. distances=[2,1,3,0], argsort的返回值为[3，1，0，2]
    sortedDistIndicies = distance.argsort()
    classCount = {}
    # 2.选择距离最小的k个点
    for i in range(k):
        # sortedDistIndicies[i] 为第i个点的索引值，voteIlibel为该样本索引下对应标签
        voteIlibel = labels[sortedDistIndicies[i]]
        # 该字典的key值为voteIlibel标签
        # classCount.get(voteIlibel, 0)为从classCount字典中寻找key：voteIlibel有则返回对应的value值无则返回0（可自定义）
        classCount[voteIlibel] = classCount.get(voteIlibel,0) + 1   # 计算voteIlibel标签下对应的样本数量
    print(classCount)
    # 3.排序
    # classCount.items()表示以列表形式返回[(key1，value1),(key2，value2),....]
    # operator.itemgetter(1)表示用于获取哪些维的数据，参数是序号，如参数为1表示以第二个维度进行排序
    # reverse=True表示降序排列，False表示升序排列
    sorrtedClassCount = sorted(classCount.items(),
                               key=operator.itemgetter(1), reverse=True)
    # 返回最多数量的标签作为输入样本的标签
    return sorrtedClassCount[0][0]

if __name__ == '__main__':
    group, labels = createDataSet()
    x = classify0([0,0], group, labels, 3)
    print(x)




