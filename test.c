#include <stdio.h>
#include <stdlib.h>

long optimize_lr_c(double *arr,long idx_start,long idx_end,long len_lr_diff,long double s_rl_diff,long double err_min)//,long double values[])
{
   //long double s_rl_diff = values[0];
   //long double err_min = values[1];
   long idx_min = idx_start-1;
   long  idx = idx_start;
   long double new_err = 0;
   
   //printf("%f, %f %ld\n",arr[0],arr[1],idx_end);
   for(idx=idx_start;idx<idx_end;idx++)
   {
      len_lr_diff += 2;
      s_rl_diff -= 2*arr[idx];
      new_err = arr[idx+1]*len_lr_diff + s_rl_diff;
      
      if(new_err < err_min)
      {
         err_min = new_err;
         idx_min = idx;
      }
      //printf("In C %ld\n",idx_min);
   }
   
   return idx_min;
}



/*
for idx in range(idx_start, idx_end):

     

        len_lr_diff +=2

        s_rl_diff -= 2*arr[idx]

        # s_left+=arr[idx]

        # s_right-=arr[idx]

        new_err = arr[idx+1]*len_lr_diff + s_rl_diff



        if new_err < err_min:

            err_min = new_err

            idx_min = idx
*/

long optimize_rl_c(double *arr,long idx_start,long idx_end,long double err,long double err_min,long double arr_last)//,long double values[])
{
   //long double s_rl_diff = values[0];
   //long double err_min = values[1];
   long idx_min = idx_start-1;
   long  idx = idx_start;
   long double new_err = 0;
   
   //printf("%f, %f %ld\n",arr[0],arr[1],idx_end);
   for(idx=idx_start;idx<idx_end;idx++)
   {
      
      
      err += 2*arr[idx] - arr_last;
      
      if(err < err_min)
      {
         err_min = err;
         idx_min = idx;
      }
      //printf("In C %ld\n",idx_min);
   }
   
   return idx_min;
}
/*
for idx in range(idx_start, idx_end):

        # len_r-=1

        # sum_right-=arr_last

        # s_lr_diff += 2*arr[idx]

        err+=2*arr[idx]-arr_last



        if err < err_min:

            err_min = err

            idx_min = idx

   

    return idx_min
*/