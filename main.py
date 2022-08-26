
#%%

# 给GA的特殊优待：

#%%
# 开始遗传算法+FF      参考代码：  https://blog.csdn.net/tangshishe/article/details/116505170

import time     ## time

import random
#生成了item和itemlist
import time     ## time

import pandas as pd
import numpy as np
import random
import math


GAtimelist=[]
GAfitlist=[]

for sumitem in range(5,30):
    item = []
    a = 0
    b = 0
    i = 0
    while i < sumitem:  # this is the Generateed items, and 100 is the number of the items generated
        a = random.randint(1, 100)
        item.append(a)
        a = 0
        i = i + 1
    # print(item)
    crossitem = item




    #%%

    # 以上已经生成了种群，popgroup， 现在开始研究下面的问题
    # 开始遗传算法+FF      参考代码：  https://blog.csdn.net/tangshishe/article/details/116505170







    def tournament_select(pops, popsize, fit, tournament_size):
        new_pops = []   # 是不是新增染色体啊？？？
        while len(new_pops) < len(pops):
            tournament_list = random.sample(range(0, popsize), tournament_size)  # pops就是popgroup，，popsize是选取的集群的大小，然后 tournament_size就是几个里面选,等于4
            # ，tournament_list就是选出来的四个染色体对应的序号
            tournament_fit = [fit[i] for i in tournament_list]     # tournament_list每一个染色体计算fit值
            # 转化为df方便索引
            tournament_df = pd.DataFrame([tournament_list, tournament_fit]).transpose().sort_values(by=1).reset_index(
                drop=True)  # 是按照fit值排序了
            new_pop = pops[int(tournament_df.iloc[-1, 0])]  # c=tournament_df.iloc[-1, 0] ,, c就是最高分(fit)的染色体的序号
      ## 这里有一个问题，就是list问题

            new_pops.append(new_pop)

        return new_pops   # 唯一的作用，就是从pops里面抽取popsize个表现最好的染色体，，也就是new_pops




    def crossover(popsize, parent1_pops, parent2_pops, pc):
        child_pops = []
        for i in range(popsize):
            parent1 = parent1_pops[i]
            parent2 = parent2_pops[i]   # 为什么parent1_pops，parent2_pops已经划分好了？两个列表了？？ 后面肯定有生成方式，，
            # 后面回来了，这两个就是分别形成的两个大小为popsize的数组


            child = [-1 for i in range(len(parent1))]  # 初始化成 -1 ，位数同染色体位数

            if random.random() >= pc:   # 不交叉验证，就copy然后乱序
                child = parent1.copy()  # 随机生成一个
                random.shuffle(child)     # 随意乱序，仅仅如此

            else:
                start_pos = random.randint(0, len(parent1) - 1)
                end_pos = random.randint(0, len(parent1) - 1)
                if start_pos > end_pos: start_pos, end_pos = end_pos, start_pos
                child[start_pos:end_pos + 1] = parent1[start_pos:end_pos + 1].copy()
                child[0:start_pos] = parent2[0:start_pos].copy()
                child[end_pos + 1:] = parent2[end_pos + 1:].copy()

            child_pops.append(child)
        return child_pops  # 这个就是染色体



    def mutate(populations, pm, boxNum):
        """
        基本位变异
        输入：populations-种群,pm-变异概率
        返回：变异后种群-population_after_mutate

        """
        population_after_mutate = []
        for i in range(len(populations)):
            pop = populations[i].copy()
            for i in range(len(pop)):
                if random.random() < pm:
                    randomList = list(range(1, boxNum + 1))
                    randomList.remove(pop[i])
                    pop[i] = random.sample(randomList, 1)[0]  # Randomly generate another box 随机生成另外一个箱子

            population_after_mutate.append(pop)  # pop是新的染色体了，但是原来的还在好像

        return population_after_mutate






    def package_calFitness(cargo_df, pop, max_v, boxNum):   # 爆仓检测，很重要
        '''
        输入：cargo_df-货物信息,pop-个体,max_v-箱子容积,max_m-箱子在载重
        输出：适应度-fit，boxes-解码后的个体
        '''
        boxes = [[] for i in range(boxNum)]
        v_sum = [0] * boxNum  # 计算每一个箱子的载重
        # m_sum = [0] * boxNum

        for j in range(len(pop)):   # pop是一个染色体
            box_id = int(pop[j]) - 1 # 从0开始记，才是box_id
            v_j = cargo_df[cargo_df['itemnumber'] == j]['weight'].iloc[0]   # 提取j号物品的体积

            boxes[box_id].append(j)  #二维数组， boxes里面的box_id位箱子（原本是[])， 装进了 j，j对应的是物品号码，这个就是我FF的solution
            v_sum[box_id] += v_j   # v_sum是一个一维数组，分别装着各个箱子的已占有空间
            # m_sum[box_id] += m_j

        num = 0  # 计数
        for i in range(boxNum):  # 所有的箱子走一遍，检测有没有爆仓，，这个boxNum 就是箱子数目，，这个是遍历循环算法
            if (v_sum[i] <= max_v) :   # max_v 就是V ，就是箱子容积
                num += 1
            else:
                break   # 跳出循环
        if num == boxNum:  # 到头了结束了，计算该解的方差
            # fit = 100 / (np.var(v_sum))  #
            fit = sum(v_sum) / max_v / boxNum   # 这里直接这么用会报错，因为这种算法得把v_sum转换为np矩阵，这是必须的，切记

            # 构造方差和的倒数，100只是让结果数值显得大一点，懂了！！ 肯定是每一个箱子总数越接近越好，方差大说明空缺大，倒数相反
        else:
            fit = -np.var(v_sum)    # 爆仓了就无法达到num == boxNum，这个解毫无意义，那么分数变成负数，直接不入流了，垃圾解，，这应该就是唯一的判断机制了

            # m_sum[box_id] += m_j

        return round(fit, 4), boxes, v_sum  # fit 四舍五入到小数点后四位
        # boxes就是我前几天那个解，很简单的，，然后v_sum就是一个一维数组，里面是每一个箱子已经占有的空间









    def package_GA(cargo_df, generations, popsize, tournament_size, pc, pm, max_v, boxNum):
        # 初始化种群

        fitlist=[]

        cargo_list = list(cargo_df['itemnumber'])
        pops = [[random.randint(1, boxNum) for i in range(len(cargo_list))] for j in range(popsize)]
        # 种群初始化，，就是用随机数生成popsize个随机解，垃圾解一大堆的
        # 不对的，这是在摆烂（垃圾解太多了），我有FF不用这个，，ff的结论放在一个list就是这个东西


        fit, boxes = [-1] * popsize, [-1] * popsize    # 每个解有一个 fit， 有一个boxes
        v_sum = [-1] * popsize   # 这个是每一个染色体的总数，popsize就是规定的种群数量，不用质疑

        for i in range(popsize):
            fit[i], boxes[i], v_sum[i] = package_calFitness(cargo_df, pops[i], max_v, boxNum)

        best_fit = max(fit)       # 看回前面的fit函数
        best_pop = pops[fit.index(max(fit))].copy()      # best_fit对应的best_pop
        best_box = boxes[fit.index(max(fit))].copy()    # 每一个解是一个装箱方案，然后该装箱方案下的 best_box ，也是一个数组
        best_vsum = v_sum[fit.index(max(fit))].copy()

        if best_fit == 1: return best_pop  # 1说明除最后一辆车都装满，已是最优解   除最后一辆车？是的，上一行代码的index 实现了这一个功能！！！




        iter = 0  # 迭代计数

        while iter < generations:
            pops1 = tournament_select(pops, popsize, fit, tournament_size)
            # pops：是[ [1,3,2,55..知道物品数], [],[]...一共有popsize个元素]，，这个popsize也是我在FF代码可以调整的
            # popsize种群大小，数值，，，fit一维数组，记录着pops每一个解的fit值 ，

            pops2 = tournament_select(pops, popsize, fit, tournament_size)   # 看回前面，，然后理解new_pops
            new_pops = crossover(popsize, pops1, pops2, pc)
            new_pops = mutate(new_pops, pm, boxNum)
            iter += 1



            new_fit, new_boxes = [-1] * popsize, [-1] * popsize  # 初始化，记录防爆仓函数数值
            newv_sum = [-1] * popsize

            for i in range(popsize):  # 防爆仓
                new_fit[i], new_boxes[i], newv_sum[i] = package_calFitness(cargo_df, new_pops[i], max_v,
                                                                                        boxNum)  # 计算适应度
            for i in range(len(pops)):
                if fit[i] < new_fit[i]:
                    pops[i] = new_pops[i]  #有更好的，全换，子代换掉父代
                    fit[i] = new_fit[i]
                    boxes[i] = new_boxes[i]
                    v_sum[i] = newv_sum[i]

            if best_fit < max(fit):  # 保留历史最优
                best_fit = max(fit)
                best_pop = pops[fit.index(max(fit))].copy()
                best_box = boxes[fit.index(max(fit))].copy()
                best_vsum = v_sum[fit.index(max(fit))].copy()

            fitlist.append((best_fit))

            # print("第", iter, "代适应度最优值：", best_fit)
        return best_pop, best_fit, best_box, best_vsum,fitlist




    GAstart=time.time()


    if __name__ == '__main__':
        # 数据
        num = list(range(len(item)))  # 货物编号 ，同一开始的 item 生成位数
        volumns = item  # 体积
        cargo_df = pd.DataFrame({'itemnumber': num, "weight": volumns})  # 简单一步就完成了物品从list 转换成pandas
        V = 100
        generations = 100
        popsize = 40
        tournament_size = 4
        pc = 0.9
        pm = 0.1
        boxNum = math.ceil(cargo_df.loc[:, "weight"].sum() / V)
        while True:
            pop, fit, box, v_list,fitlist = package_GA(cargo_df, generations, popsize, tournament_size, pc, pm, V,
                                                       boxNum)
            if fit > 0:
                break
            else:
                boxNum += 1

        # print("最优解：", box)
        # print("箱子容积分别为：", v_list)
        # print(fitlist)    # 我自己加的 fitlist

    GAend=time.time()     ## time
    GAtime=GAend-GAstart
    GAtimelist.append(GAtime)

    GAfitlist.append(fit)

    # print('Running time: %s Seconds'%(end-start))      ## time
#%%
print(GAtimelist)
print(GAfitlist)





#%%

import numpy as np
import matplotlib.pyplot as plt


sumitemlist=[]
for i in range(5,30):
    sumitemlist.append(i)



print(sumitemlist)

x1=sumitemlist
y1=GAfitlist
l1=plt.plot(x1,y1,'g--',label='GAmaxfit')
# l2=plt.plot(x2,y2,'g--',label='GAFFtime')

plt.plot(x1,y1,'g+-')
plt.title('GAmaxfit')
plt.xlabel('itemnumber')
plt.ylabel('maxfit')
plt.legend()
plt.show()



