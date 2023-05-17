%% MTMF

%h5disp('C:/Research/stratified/MTMF/snapshots/snapshots_s1/snapshots_s1_p0.h5')

mtmf_A2 = h5read('C:/Research/stratified/MTMF/snapshots/snapshots_s1/snapshots_s1_p0.h5','/tasks/A_square').r;
mtmf_A = sqrt(mtmf_A2);

mtmf_b = h5read('C:/Research/stratified/MTMF/snapshots/snapshots_s1/snapshots_s1_p0.h5','/tasks/b').r;
mtmf_b1 = h5read('C:/Research/stratified/MTMF/snapshots/snapshots_s1/snapshots_s1_p0.h5','/tasks/bw');
mtmf_b2 = h5read('C:/Research/stratified/MTMF/snapshots/snapshots_s1/snapshots_s1_p0.h5','/tasks/b2');
mtmf_b1_2 = h5read('C:/Research/stratified/MTMF/snapshots/snapshots_s1/snapshots_s1_p0.h5','/tasks/b1_2');

mtmf_b1 = mtmf_b1.r + 1i*mtmf_b1.i;
mtmf_b2 = mtmf_b2.r + 1i*mtmf_b2.i;
mtmf_b1_2 = mtmf_b1_2.r + 1i*mtmf_b1_2.i;

mtmf_u = h5read('C:/Research/stratified/MTMF/snapshots/snapshots_s1/snapshots_s1_p0.h5','/tasks/u').r;
mtmf_psi = h5read('C:/Research/stratified/MTMF/snapshots/snapshots_s1/snapshots_s1_p0.h5','/tasks/Psi');
mtmf_psi2 = h5read('C:/Research/stratified/MTMF/snapshots/snapshots_s1/snapshots_s1_p0.h5','/tasks/Psi2');
mtmf_psi1_2 = h5read('C:/Research/stratified/MTMF/snapshots/snapshots_s1/snapshots_s1_p0.h5','/tasks/Psi1_2');

mtmf_psi = mtmf_psi.r + 1i*mtmf_psi.i;
mtmf_psi2 = mtmf_psi2.r + 1i*mtmf_psi2.i;
mtmf_psi1_2 = mtmf_psi1_2.r + 1i*mtmf_psi1_2.i;



mtmf_psi = mtmf_psi*diag(mtmf_A) + 0*mtmf_psi1_2*diag(mtmf_A)^3;
mtmf_b1 = mtmf_b1*diag(mtmf_A) + 0*mtmf_b1_2*diag(mtmf_A)^3;

mtmf_psi2 = mtmf_psi2*diag(mtmf_A)^2;
mtmf_b2 = mtmf_b2*diag(mtmf_A)^2;


M = size(mtmf_psi,1);
mtmf_u1 = trigtech.coeffs2vals(diag(1i*(-M/2:M/2-1))*trigtech.vals2coeffs(mtmf_psi));
mtmf_u2 = trigtech.coeffs2vals(diag(1i*(-M/2:M/2-1))*trigtech.vals2coeffs(mtmf_psi2));

mtmf_zz = h5read('C:/Research/stratified/MTMF/snapshots/snapshots_s1/snapshots_s1_p0.h5','/scales/z/1.0');
mtmf_kk = h5read('C:/Research/stratified/MTMF/snapshots/snapshots_s1/snapshots_s1_p0.h5','/tasks/k').r;

mtmf_w1 = 1i*mtmf_psi*diag(mtmf_kk);
mtmf_w2 = 2i*mtmf_psi2*diag(mtmf_kk);

N = 100;
eps = 1;sqrt(0.02);


k = mtmf_kk(end);
mtmf_xx = linspace(-pi/k,pi/k,N);



figure;
nw = [mtmf_xx(end), mtmf_zz(1)];
se = [mtmf_xx(1), mtmf_zz(end)];
%imagesc(nw,se,(1/eps)*real(mtmf_w1(:,end)*exp(1i*k*mtmf_xx))+real(mtmf_w2(:,end)*exp(2i*k*mtmf_xx)))
imagesc(nw,se,mtmf_u(:,end)*ones(1,N) + eps*real(mtmf_u1(:,t)*exp(1i*k*mtmf_xx))+eps^2*real(mtmf_u2(:,end)*exp(2i*k*mtmf_xx)))
%imagesc(nw,se,mtmf_b(:,end)*ones(1,N) + eps*real(mtmf_b1(:,end)*exp(1i*k*mtmf_xx))+eps^2*real(mtmf_b2(:,end)*exp(2i*k*mtmf_xx)))
title("MTMF")

%% MTQL

%h5disp('C:/Research/stratified/MTMF/snapshots/snapshots_s1/snapshots_s1_p0.h5')

mtql_A2 = h5read('C:/Research/stratified/MTQL/snapshots/snapshots_s1/snapshots_s1_p0.h5','/tasks/A_square').r;
mtql_A = sqrt(mtql_A2);

mtql_u = h5read('C:/Research/stratified/MTQL/snapshots/snapshots_s1/snapshots_s1_p0.h5','/tasks/u').r;
mtql_psi = h5read('C:/Research/stratified/MTQL/snapshots/snapshots_s1/snapshots_s1_p0.h5','/tasks/Psi');
mtql_b = h5read('C:/Research/stratified/MTQL/snapshots/snapshots_s1/snapshots_s1_p0.h5','/tasks/b').r;
mtql_b1 = h5read('C:/Research/stratified/MTQL/snapshots/snapshots_s1/snapshots_s1_p0.h5','/tasks/bw');

mtql_psi = mtql_psi.r + 1i*mtql_psi.i;
mtql_b1 = mtql_b1.r + 1i*mtql_b1.i;

mtql_psi = mtql_psi*diag(mtql_A);


M = size(mtql_psi,1);
mtql_u1 = trigtech.coeffs2vals(diag(1i*(-M/2:M/2-1))*trigtech.vals2coeffs(mtql_psi));


mtql_zz = h5read('C:/Research/stratified/MTQL/snapshots/snapshots_s1/snapshots_s1_p0.h5','/scales/z/1.0');
mtql_kk = h5read('C:/Research/stratified/MTQL/snapshots/snapshots_s1/snapshots_s1_p0.h5','/tasks/k').r;

mtql_w = 1i*mtql_psi*diag(mtql_kk);

N = 100;
eps = 1%sqrt(0.02);


k = mtql_kk(end);
mtql_xx = linspace(-pi/k,pi/k,N);



figure;
nw = [mtql_xx(end), mtql_zz(1)];
se = [mtql_xx(1), mtql_zz(end)];
imagesc(nw,se,mtql_u(:,end)*ones(1,N) + eps*real(mtql_u1(:,end)*exp(1i*k*mtql_xx)))
%imagesc(nw,se,eps^(-1)*real(mtql_w(:,end)*exp(1i*k*mtql_xx)))
%imagesc(nw,se,(0*mtql_zz+mtql_b(:,end))*ones(1,N) + eps*real(mtql_b1(:,end)*exp(1i*k*mtql_xx)))

title("MTQL")

%% DNS

%h5disp('C:/Research/stratified/MTMF/snapshots/snapshots_s1/snapshots_s1_p0.h5')

dns_u = h5read('C:/Research/stratified/DNS/snapshots/snapshots_s1/snapshots_s1_p0.h5','/tasks/u');
dns_b = h5read('C:/Research/stratified/DNS/snapshots/snapshots_s1/snapshots_s1_p0.h5','/tasks/b');
dns_w = h5read('C:/Research/stratified/DNS/snapshots/snapshots_s1/snapshots_s1_p0.h5','/tasks/w');

M = size(dns_u,1);
N = size(dns_u,2);


dns_zz = h5read('C:/Research/stratified/DNS/snapshots/snapshots_s1/snapshots_s1_p0.h5','/scales/kz');
dns_xx = h5read('C:/Research/stratified/DNS/snapshots/snapshots_s1/snapshots_s1_p0.h5','/scales/kx');



figure;
nw = [mtql_xx(end), mtql_zz(1)];
se = [mtql_xx(1), mtql_zz(end)];
%imagesc(nw,se,dns_b(:,:,end))
%imagesc(nw,se,dns_w(:,:,end))
imagesc(nw,se,dns_u(:,:,end))
%imagesc(nw,se,(mtql_zz+mtql_b(:,end))*ones(1,N) + eps*real(mtql_b1(:,end)*exp(-1i*k*mtql_xx)))

title("DNS")