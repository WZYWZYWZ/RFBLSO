function wk = sparse_bls(A,b,lam,itrs)%A=A1(60000*10)第一次生成的随机矩阵;%b=H1(60000*785);输入的数据%lam 误差范围%itrs 迭代次数
AA = (A') * A;%这一步为啥要这么做（特征的进一步融合）
m = size(A,2);%取新特征矩阵的列
n = size(b,2);%取原始数据矩阵数据的列
x = zeros(m,n);%生成一个矩阵，里面全0（10*785）
wk = x; %(10*785)
ok=x;uk=x;
L1=eye(m)/(AA+eye(m));%eye(m)是生成10*10的单位阵，L1是进一步生成带有特征的矩阵
L2=L1*A'*b;%(10*785)

for i = 1:itrs,%（其实wk大约在18就可以结束了，为什么要设置为50）
    tempc=ok-uk;
  ck =  L2+L1*tempc;
 ok=shrinkage(ck+uk, lam);%调用了shrinkage函数
 uk=uk+(ck-ok);%更新uk
 wk=ok;%将求出的新的特征节点的系数输出
end
end
function z = shrinkage(x, kappa)%为什么kappa=lam中lam要设定为这个值
    z = max( x - kappa,0 ) - max( -x - kappa ,0);
end

% % toc

