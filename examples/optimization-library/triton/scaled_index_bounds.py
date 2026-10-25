"""Allocation-free signed-index certificate; all arithmetic uses Python integers."""
LIMIT=(1<<31)-1


def index_audit(m,n,k,a_stride,b_stride,c_stride,storage_offsets=(0,0,1)):
    if any(type(x) is not int or x<=0 for x in (m,n,k)):
        raise ValueError('positive integer dimensions required')
    for strides in (a_stride,b_stride,c_stride):
        if not isinstance(strides,(list,tuple)) or len(strides)!=2 or any(type(x) is not int or x<=0 for x in strides):
            raise ValueError('positive physical strides required')
    if len(storage_offsets)!=3 or any(type(x) is not int or x<0 for x in storage_offsets):
        raise ValueError('nonnegative storage offsets required')
    mp=((m+15)//16)*16;np=((n+63)//64)*64
    # Source offsets_k and both pointers increment after the final dot as well.
    kp=((k+127)//128)*128+127
    am,ak=a_stride;bk,bn=b_stride;cm,cn=c_stride
    products=dict(a_row=(mp-1)*am,a_k=kp*ak,b_k=kp*bk,b_col=(np-1)*bn,
                  c_row=(mp-1)*cm,c_col=(np-1)*cn)
    bounds={**products,'a_sum':products['a_row']+products['a_k'],
            'b_sum':products['b_k']+products['b_col'],'c_sum':products['c_row']+products['c_col'],
            'grid_programs':(mp//16)*(np//64),'pid_m_product':mp-16,'pid_n_product':np-64,
            'bias_index':np-1,'scalar_scale_index':0,'k_increment':128,
            'a_pointer_increment':128*ak,'b_pointer_increment':128*bk,
            'M':m,'N':n,'K':k,'N_cdiv_add':n+63,'K_cdiv_add':k+127,'K_after_final_increment':kp}
    for prefix,strides in [('a',a_stride),('b',b_stride),('c',c_stride)]:
        for i,stride in enumerate(strides):bounds[prefix+'_stride_'+str(i)]=stride
    for name,size,offset in zip(('a','b','c'),(1,1,4),storage_offsets):
        bounds[name+'_byte_extent']=(bounds[name+'_sum']+offset+1)*size-1
    bounds.update(bias_byte_extent=np*4-1,scale_byte_extent=3)
    unsafe={key:value for key,value in bounds.items() if value>LIMIT}
    return dict(fits_int32=not unsafe,bounds=bounds,unsafe=unsafe,padded=[mp,np,kp],
                note='Nonnegative products/sums bound every intermediate and masked padded lane; pointer bases stay64bit. Byte extents include storage offset and final byte.')
