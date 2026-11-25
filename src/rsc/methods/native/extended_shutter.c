/* Wider search, symmetric visible footprints, and four overlapping time intervals. */
#include "block_shutter.c"

enum { EXT_FEATURES=4*PAIRS*2, EXT_NODES=CELLS*2 };
typedef struct {Pyramid base; uint8_t tiny[32*24];} WidePyramid;
typedef struct {WidePyramid images[2];int slot,ready;} WideMotion;

static void wide_thumbnail(const uint8_t *img,WidePyramid *p) {
    thumbnail_pyramid(img,&p->base);
    for(int y=0;y<24;++y) for(int x=0;x<32;++x) {
        const uint8_t *a=p->base.coarse+(y*2)*64+2*x;
        p->tiny[y*32+x]=(uint8_t)((a[0]+a[1]+a[64]+a[65]+2)>>2);
    }
}
static int sad16(const uint8_t *a,const uint8_t *b,int pitch,int count,int step) {
    __m128i sum=_mm_setzero_si128();
    for(int n=0;n<count;++n,a+=step*pitch,b+=step*pitch)
        sum=_mm_add_epi64(sum,_mm_sad_epu8(_mm_loadu_si128((const __m128i*)a),_mm_loadu_si128((const __m128i*)b)));
    return _mm_cvtsi128_si32(_mm_add_epi64(sum,_mm_srli_si128(sum,8)));
}
static int clip_range(int x,int lo,int hi) {return x<lo?lo:(x>hi?hi:x);}
typedef struct {int x,y;} Position;
static Position footprint(int cx,int cy,int dx,int dy,int width,int height,int span,int lines,int step,int margin) {
    int lo_x=(dx<0?-dx:0)+margin, lo_y=(dy<0?-dy:0)+margin;
    int hi_x=width-span-(dx>0?dx:0)-margin,hi_y=height-1-(lines-1)*step-(dy>0?dy:0)-margin;
    Position p={clip_range(cx-span/2-dx/2,lo_x,hi_x),
                clip_range(cy-((lines-1)*step)/2-dy/2,lo_y,hi_y)};
    return p;
}
static void search16(const uint8_t *a,const uint8_t *b,int w,const Position *regions,int regions_n,
                     int lines,int step,int radius,int penalty,int *dx,int *dy) {
    int bx=*dx,by=*dy,best=2147483647;
    for(int v=-radius;v<=radius;++v) for(int u=-radius;u<=radius;++u) {
        int cost=penalty*(u*u+v*v);
        for(int j=0;j<regions_n;++j) {
            int i=regions[j].x+w*regions[j].y;
            cost+=sad16(a+i,b+i+w*(by+v)+bx+u,w,lines,step);
        }
        if(cost<best) {best=cost;*dx=bx+u;*dy=by+v;}
    }
}
static void search32(const uint8_t *a,const uint8_t *b,const Position p,int lines,int step,
                     int radius,int penalty,int *dx,int *dy) {
    int bx=*dx,by=*dy,best=2147483647,i=p.x+TW*p.y;
    for(int v=-radius;v<=radius;++v) for(int u=-radius;u<=radius;++u) {
        int cost=block_sad(a+i,b+i+(by+v)*TW+bx+u,TW,lines,step)+penalty*(u*u+v*v);
        if(cost<best) {best=cost;*dx=bx+u;*dy=by+v;}
    }
}

static void wide_pair(const WidePyramid *a,const WidePyramid *b,float out[PAIRS][2]) {
    int dx=0,dy=0;
    Position tiny={8,8};
    search16(a->tiny,b->tiny,32,&tiny,1,8,1,8,1,&dx,&dy);
    dx*=2;dy*=2;
    Position regions[3];
    for(int j=0;j<3;++j)
        regions[j]=footprint(16+j*16,(j==1?32:17),dx,dy,CW,CH,16,6,2,2);
    search16(a->base.coarse,b->base.coarse,CW,regions,3,6,2,2,5,&dx,&dy);
    dx*=2;dy*=2;
    Position center=footprint(64,48,dx,dy,TW,TH,32,10,3,3);
    search32(a->base.fine,b->base.fine,center,10,3,2,6,&dx,&dy);
    subpixel(a->base.fine,b->base.fine,center.x,center.y,dx,dy,10,3,out[0]);
    for(int k=0;k<CELLS;++k) {
        int cx=28+(k%3)*36,cy=29+(k/3)*39;
        int lx=(int)roundf(dx*.5f),ly=(int)roundf(dy*.5f);
        Position coarse=footprint(cx/2,cy/2,lx,ly,CW,CH,16,8,1,3);
        search16(a->base.coarse,b->base.coarse,CW,&coarse,1,8,1,3,16,&lx,&ly);
        lx*=2; ly*=2;
        Position fine=footprint(cx,cy,lx,ly,TW,TH,32,8,2,3);
        search32(a->base.fine,b->base.fine,fine,8,2,2,12,&lx,&ly);
        subpixel(a->base.fine,b->base.fine,fine.x,fine.y,lx,ly,8,2,out[k+1]);
    }
}
EXPORT WideMotion *wide_new(void) {return calloc(1,sizeof(WideMotion));}
EXPORT void wide_delete(WideMotion *m) {free(m);}
EXPORT void wide_read(WideMotion *m,const uint8_t *a,const uint8_t *b,float *out) {
    if(!m->ready) wide_thumbnail(a,&m->images[m->slot]);
    wide_thumbnail(b,&m->images[1-m->slot]);
    wide_pair(&m->images[m->slot],&m->images[1-m->slot],(float (*)[2])out);
    m->slot=1-m->slot;m->ready=1;
}

/* Independent matches on a 6 x 4 grid, expressed as residuals from the coarser field. */
EXPORT void detail_pair(WideMotion *m,const float *flow,float *residual) {
    const uint8_t *a=m->images[1-m->slot].base.fine,*b=m->images[m->slot].base.fine;
    for(int cell=0;cell<24;++cell) {
        float x=(cell%6+.5f)*128.f/6.f,y=(cell/6+.5f)*24.f;
        float u=fminf(2.4f,fmaxf(-.4f,(x-28.f)/36.f)),v=fminf(1.4f,fmaxf(-.4f,(y-29.f)/39.f));
        int col=u>1?1:0;u-=col;
        float ref[2];
        for(int d=0;d<2;++d) {
            const float *z=flow+2+col*2+d;
            float top=z[0]+u*(z[2]-z[0]),bottom=z[6]+u*(z[8]-z[6]);
            ref[d]=top+v*(bottom-top);
        }
        int dx=(int)roundf(ref[0]),dy=(int)roundf(ref[1]);
        Position patch=footprint((int)x,(int)y,dx,dy,TW,TH,16,7,2,3);
        search16(a,b,TW,&patch,1,7,2,2,7,&dx,&dy);
        int aa=180,bb=180,ab=0,ar=0,br=0;
        for(int j=0;j<7;++j) {
            int i=(patch.y+2*j)*TW+patch.x,k=i+dy*TW+dx;
            for(int h=1;h<15;h+=2) {
                int gx=b[k+h+1]-b[k+h-1]+a[i+h+1]-a[i+h-1];
                int gy=b[k+h+TW]-b[k+h-TW]+a[i+h+TW]-a[i+h-TW];
                int e=a[i+h]-b[k+h]; aa+=gx*gx;bb+=gy*gy;ab+=gx*gy;ar+=gx*e;br+=gy*e;
            }
        }
        float det=(float)aa*bb-(float)ab*ab;
        residual[2*cell]=dx-ref[0]+fminf(.85f,fmaxf(-.85f,4*((float)bb*ar-(float)ab*br)/det));
        residual[2*cell+1]=dy-ref[1]+fminf(.85f,fmaxf(-.85f,4*((float)aa*br-(float)ab*ar)/det));
    }
}
EXPORT void block_extra(BlockRestorer *s,const float *extra) {
    s->dense=extra!=NULL;if(extra)memcpy(s->extra,extra,sizeof(s->extra));
}

typedef struct {
    BlockRestorer *render;
    WideMotion motion;
    float history[4][PAIRS*2],following[PAIRS*2],weights[EXT_NODES][EXT_FEATURES+1];
    int ready;
    float center[48],precision[48][48],inverse_root[48],cut;
    const TreeNode *nodes;
    const int32_t *roots;
    int tree_count;
    float forest_bias[2],forest_gain;
    float fine[4][48],next_fine[48],detail_weights[2][25];
    int detail;
    const TreeNode *fine_nodes;
    const int32_t *fine_roots;
    int fine_count;
    float fine_bias[2],fine_gain;
    const uint8_t *a,*b;
    PTP_WORK work;
} Extended;
EXPORT void ext_grid(Extended *s,const float *weights) {
    s->detail=1;memcpy(s->detail_weights,weights,sizeof(s->detail_weights));
}
EXPORT void ext_grid_trees(Extended *s,const TreeNode *nodes,const int32_t *roots,int count,const float *bias,float gain) {
    s->fine_nodes=nodes;s->fine_roots=roots;s->fine_count=count;s->fine_gain=gain;
    memcpy(s->fine_bias,bias,sizeof(s->fine_bias));
}
static void dense_prediction(Extended *s) {
    s->render->dense=1;
    for(int p=0;p<24;++p) {
        int col=p%6,row=p/6,left=p-(col>0),right=p+(col<5);
        int top=p-6*(row>0),bottom=p+6*(row<3);
        float features[25],result[2]={0};
        for(int t=0;t<4;++t) for(int d=0;d<2;++d) {
            features[t*6+d]=s->fine[t][p*2+d];
            features[t*6+2+d]=.5f*(s->fine[t][left*2+d]+s->fine[t][right*2+d]);
            features[t*6+4+d]=.5f*(s->fine[t][top*2+d]+s->fine[t][bottom*2+d]);
        }
        features[24]=1;
        for(int d=0;d<2;++d)
            for(int k=0;k<25;++k)result[d]+=s->detail_weights[d][k]*features[k];
        if(s->fine_count) {
            result[0]+=s->fine_bias[0];result[1]+=s->fine_bias[1];
            for(int i=0;i<s->fine_count;++i) {
                const TreeNode *node=s->fine_nodes+s->fine_roots[i];
                while(node->left>=0) {
                    int child=(int)(features[(int)node->feature]<=node->threshold?node->left:node->right);
                    node=s->fine_nodes+child;
                }
                result[0]+=node->value[0];result[1]+=node->value[1];
            }
            result[0]*=s->fine_gain;result[1]*=s->fine_gain;
        }
        for(int d=0;d<2;++d)s->render->extra[row][col][d]=result[d];
    }
}
EXPORT void ext_trees(Extended *s,const TreeNode *nodes,const int32_t *roots,int n,const float *bias,float gain) {
    s->nodes=nodes;s->roots=roots;s->tree_count=n;s->forest_gain=gain;
    memcpy(s->forest_bias,bias,sizeof(s->forest_bias));
}
static void local_trees(Extended *s,float (*motion)[PAIRS*2],float *v) {
    float common[4][2]={{0}};
    for(int t=0;t<4;++t) for(int p=0;p<6;++p) for(int d=0;d<2;++d)
        common[t][d]+=motion[t][2*(p+1)+d]/6.f;
    for(int p=0;p<6;++p) {
        float f[28],out[2]={s->forest_bias[0],s->forest_bias[1]};
        f[0]=v[p*2];f[1]=v[2*p+1];f[2]=p%3-1;f[3]=2*(p/3)-1;
        for(int t=0;t<4;++t) for(int d=0;d<2;++d) {
            f[4+t*6+d]=motion[t][2*(p+1)+d]-f[d];
            f[6+t*6+d]=motion[t][d]-f[d];
            f[8+t*6+d]=common[t][d]-f[d];
        }
        for(int i=0;i<s->tree_count;++i) {
            const TreeNode *node=s->nodes+s->roots[i];
            while(node->left>=0) {
                int k=(int)(f[(int)node->feature]<=node->threshold?node->left:node->right);
                node=s->nodes+k;
            }
            out[0]+=node->value[0];out[1]+=node->value[1];
        }
        for(int d=0;d<2;++d)v[2*p+d]=(f[d]+out[d])*s->forest_gain;
    }
}
EXPORT void ext_gate(Extended *s,const float *mu,const float *precision,float threshold) {
    memcpy(s->center,mu,sizeof(s->center));memcpy(s->precision,precision,sizeof(s->precision));
    for(int i=0;i<48;++i) s->inverse_root[i]=1.f/sqrtf(s->precision[i][i]);
    s->cut=threshold;
}
static float median6(const float *flow,int d) {
    float v[6];
    for(int i=0;i<6;++i) {
        float x=flow[2*(i+1)+d]; int j=i;
        for(;j>0 && v[j-1]>x;--j) v[j]=v[j-1];
        v[j]=x;
    }
    return .5f*(v[2]+v[3]);
}
static void filter_motion(Extended *s,float (*f)[PAIRS*2]) {
    float z[48],scale[4];
    for(int t=0;t<4;++t) {
        float center[2]={median6(f[t],0),median6(f[t],1)};
        scale[t]=.50f+.06f*sqrtf(center[0]*center[0]+center[1]*center[1]);
        for(int i=0;i<12;++i) z[t*12+i]=(f[t][2+i]-center[i%2])/scale[t]-s->center[t*12+i];
    }
    for(int i=0;i<48;++i) {
        float sum=0;
        for(int j=0;j<48;++j) sum+=s->precision[i][j]*z[j];
        float sd=s->inverse_root[i],std=sum*sd;
        float excess=copysignf(fmaxf(0,fabsf(std)-s->cut),std)*sd;
        f[i/12][2+i%12]-=scale[i/12]*fminf(8,fmaxf(-8,excess));
    }
}
static void CALLBACK future_field(PTP_CALLBACK_INSTANCE instance,void *context,PTP_WORK work) {
    (void)instance;(void)work;
    Extended *s=context;wide_read(&s->motion,s->a,s->b,s->following);
    if(s->detail)detail_pair(&s->motion,s->following,s->next_fine);
}
EXPORT void ext_delete(Extended *s) {
    if(!s)return;
    if(s->work)CloseThreadpoolWork(s->work);
    block_delete(s->render);free(s);
}
EXPORT Extended *ext_new(const float *w,int threads,int stripes,int sampler) {
    Extended *s=calloc(1,sizeof(*s));
    if(!s)return NULL;
    float unused[CELLS][2][FEATURES+1]={0};
    s->render=block_new(threads,stripes,&unused[0][0][0]);s->work=CreateThreadpoolWork(future_field,s,NULL);
    if(!s->work || !s->render) {ext_delete(s);return NULL;}
    block_sampler(s->render,sampler);
    memcpy(s->weights,w,sizeof(s->weights));
    return s;
}
EXPORT void ext_correct(Extended *s,const uint8_t *a,const uint8_t *b,const uint8_t *c,
                        const uint8_t *d,uint8_t *out) {
    size_t pair=sizeof(s->history[0]);
    if(!s->ready) {
        wide_read(&s->motion,a,b,s->history[2]);
        if(s->detail) {
            detail_pair(&s->motion,s->history[2],s->fine[2]);
            memcpy(s->fine[0],s->fine[2],sizeof(s->next_fine));memcpy(s->fine[1],s->fine[2],sizeof(s->next_fine));
        }
        memcpy(s->history[0],s->history[2],pair);memcpy(s->history[1],s->history[2],pair);
        if(c) {
            wide_read(&s->motion,b,c,s->history[3]);
            if(s->detail)detail_pair(&s->motion,s->history[3],s->fine[3]);
        } else {
            memcpy(s->history[3],s->history[2],pair);
            if(s->detail)memcpy(s->fine[3],s->fine[2],sizeof(s->next_fine));
        }
        s->ready=1;
    } else {
        memmove(s->history[0],s->history[1],3*pair);
        memcpy(s->history[3],s->following,pair);
        if(s->detail) {memmove(s->fine[0],s->fine[1],3*sizeof(s->next_fine));memcpy(s->fine[3],s->next_fine,sizeof(s->next_fine));}
    }
    if(c && d) {s->a=c;s->b=d;SubmitThreadpoolWork(s->work);}
    else {
        memcpy(s->following,s->history[3],pair);
        if(s->detail)memcpy(s->next_fine,s->fine[3],sizeof(s->next_fine));
    }
    float velocities[EXT_NODES],corrected[4][PAIRS*2];
    float (*data)[PAIRS*2]=s->history;
    if(s->cut>0) {
        memcpy(corrected,s->history,sizeof(corrected));filter_motion(s,corrected);data=corrected;
    }
    for(int p=0;p<EXT_NODES;++p) {
        float v=s->weights[p][EXT_FEATURES];
        for(int t=0;t<4;++t)
            for(int i=0;i<PAIRS*2;++i) v+=s->weights[p][t*PAIRS*2+i]*data[t][i];
        velocities[p]=v;
    }
    if(s->tree_count)local_trees(s,data,velocities);
    if(s->detail)dense_prediction(s);
    block_render(s->render,a,out,velocities);
    if(c && d) WaitForThreadpoolWorkCallbacks(s->work,FALSE);
}
