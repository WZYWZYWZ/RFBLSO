function wk = sparse_bls(A,b,lam,itrs)%A=A1(60000*10)��һ�����ɵ��������;%b=H1(60000*785);���������%lam ��Χ%itrs ��������
AA = (A') * A;%��һ��ΪɶҪ��ô���������Ľ�һ���ںϣ�
m = size(A,2);%ȡ�������������
n = size(b,2);%ȡԭʼ���ݾ������ݵ���
x = zeros(m,n);%����һ����������ȫ0��10*785��
wk = x; %(10*785)
ok=x;uk=x;
L1=eye(m)/(AA+eye(m));%eye(m)������10*10�ĵ�λ��L1�ǽ�һ�����ɴ��������ľ���
L2=L1*A'*b;%(10*785)

for i = 1:itrs,%����ʵwk��Լ��18�Ϳ��Խ����ˣ�ΪʲôҪ����Ϊ50��
    tempc=ok-uk;
  ck =  L2+L1*tempc;
 ok=shrinkage(ck+uk, lam);%������shrinkage����
 uk=uk+(ck-ok);%����uk
 wk=ok;%��������µ������ڵ��ϵ�����
end
end
function z = shrinkage(x, kappa)%Ϊʲôkappa=lam��lamҪ�趨Ϊ���ֵ
    z = max( x - kappa,0 ) - max( -x - kappa ,0);
end

% % toc

