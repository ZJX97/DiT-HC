# source /pacific_fs/xuyunpu/app/anaconda3/bin/activate
# conda activate DiT
#
#module use /pacific_fs/HPCKit/latest/modulefiles
#module add bisheng/compiler4.1.0/bishengmodule
#module add bisheng/hmpi2.4.3/hmpi
#module add bisheng/kml25.0.0/kblas/multi
#


source /pacific_fs/anaconda3/bin/activate
conda activate DiT_DNN

module use /pacific_fs/HPCKit/25.3.30/modulefiles
module add bisheng/compiler4.1.0/bishengmodule
module add bisheng/hmpi2.4.3/hmpi
module add bisheng/kml25.0.0/kblas/multi

#export LD_LIBRARY_PATH=/pacific_ext/liujt/lib/onednn:$LD_LIBRARY_PATH

#export HOME=/pacific_ext/liujt

