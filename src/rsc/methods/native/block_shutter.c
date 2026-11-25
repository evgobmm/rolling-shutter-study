/* Streaming, thumbnail block matching and full-size rolling-shutter resampling. */
#include "scanline_native.c"

enum { TW=128, TH=96, CW=64, CH=48, COLS=3, ROWS=2, CELLS=6, PAIRS=7, FEATURES=28 };
typedef struct { uint8_t fine[TW*TH], coarse[CW*CH]; } Pyramid;
typedef struct {
    Pyramid frames[2];
    float previous[PAIRS][2], latest[PAIRS][2];
    int slot, ready;
} Matcher;

static void thumbnail_pyramid(const uint8_t *image, Pyramid *p) {
    /* Four green samples spaced across each 8x8 source footprint. */
    for (int y=0;y<TH;++y) {
        const uint8_t *a=image+((size_t)(8*y+2)*1024+2)*3+1;
        const uint8_t *b=a+4*3072;
        for (int x=0;x<TW;++x,a+=24,b+=24)
            p->fine[y*TW+x]=(uint8_t)((a[0]+a[12]+b[0]+b[12]+2)>>2);
    }
    for (int y=0;y<CH;++y) {
        const uint8_t *a=p->fine+2*y*TW, *b=a+TW;
        __m256i ones=_mm256_set1_epi8(1), rounding=_mm256_set1_epi16(2);
        for (int x=0;x<TW;x+=32) {
            __m256i sa=_mm256_maddubs_epi16(_mm256_loadu_si256((const __m256i*)(a+x)),ones);
            __m256i sb=_mm256_maddubs_epi16(_mm256_loadu_si256((const __m256i*)(b+x)),ones);
            __m256i v=_mm256_srli_epi16(_mm256_add_epi16(_mm256_add_epi16(sa,sb),rounding),2);
            _mm_storeu_si128((__m128i*)(p->coarse+y*CW+x/2),
                _mm_packus_epi16(_mm256_castsi256_si128(v),_mm256_extracti128_si256(v,1)));
        }
    }
}

/* The 32-pixel row fits one AVX2 operation. All candidate endpoints are inside the thumbnail. */
static int block_sad(const uint8_t *a,const uint8_t *b,int stride,int lines,int step) {
    __m256i sum=_mm256_setzero_si256();
    for (int i=0;i<lines;++i,a+=stride*step,b+=stride*step)
        sum=_mm256_add_epi64(sum,_mm256_sad_epu8(_mm256_loadu_si256((const __m256i*)a),
                                                   _mm256_loadu_si256((const __m256i*)b)));
    __m128i tmp=_mm_add_epi64(_mm256_castsi256_si128(sum),_mm256_extracti128_si256(sum,1));
    return _mm_cvtsi128_si32(_mm_add_epi64(tmp,_mm_srli_si128(tmp,8)));
}

static void search(const uint8_t *a,const uint8_t *b,int width,int height,int x,int y,
                   int lines,int step,int radius,int *dx,int *dy) {
    int x0=(int)fminf(width-32-x,fmaxf(-x,*dx));
    int y0=(int)fminf(height-1-y-(lines-1)*step,fmaxf(-y,*dy)),best=2147483647;
    for (int j=y0-radius;j<=y0+radius;++j) {
        if(y+j<0 || y+j+(lines-1)*step>=height) continue;
        for(int i=x0-radius;i<=x0+radius;++i) {
            if(x+i<0 || x+i+32>width) continue;
            int cost=block_sad(a+y*width+x,b+(y+j)*width+x+i,width,lines,step);
            if(cost<best) {best=cost; *dx=i; *dy=j;}
        }
    }
}

static void subpixel(const uint8_t *a,const uint8_t *b,int x,int y,int dx,int dy,
                     int lines,int step,float out[2]) {
    /* Symmetric image derivatives, one well-conditioned 2D least-squares update. */
    int aa=200,ab=0,bb=200,ar=0,br=0;
    for (int k=0;k<lines;++k) {
        int i=(y+k*step)*TW+x, j=i+dy*TW+dx;
        for(int s=1;s<31;s+=2) {
            int gx=(b[j+s+1]-b[j+s-1])+(a[i+s+1]-a[i+s-1]);
            int gy=(b[j+s+TW]-b[j+s-TW])+(a[i+s+TW]-a[i+s-TW]);
            int residual=a[i+s]-b[j+s];
            aa+=gx*gx; ab+=gx*gy; bb+=gy*gy; ar+=gx*residual; br+=gy*residual;
        }
    }
    float det=(float)aa*bb-(float)ab*ab;
    float rx=4*((float)bb*ar-(float)ab*br)/det, ry=4*((float)aa*br-(float)ab*ar)/det;
    out[0]=dx+fminf(.85f,fmaxf(-.85f,rx)); out[1]=dy+fminf(.85f,fmaxf(-.85f,ry));
}

static void pair_flow(const Pyramid *a,const Pyramid *b,float flow[PAIRS][2]) {
    int dx=0,dy=0;
    search(a->coarse,b->coarse,CW,CH,16,9,8,4,8,&dx,&dy);
    dx*=2; dy*=2;
    search(a->fine,b->fine,TW,TH,48,28,9,5,2,&dx,&dy);
    subpixel(a->fine,b->fine,48,28,dx,dy,9,5,flow[0]);
    for(int k=0;k<CELLS;++k) {
        int x=(int[]){12,48,84}[k%COLS], y=(k/COLS)?59:20;
        int lx=dx,ly=dy;
        search(a->fine,b->fine,TW,TH,x,y,7,3,3,&lx,&ly);
        /* The search enforces the patch bounds. The gradient uses one extra row. */
        int safe_y=ly;
        if(y+ly<=0) safe_y=1-y;
        if(y+ly+18>=TH-1) safe_y=TH-2-y-18;
        subpixel(a->fine,b->fine,x,y,lx,safe_y,7,3,flow[k+1]);
    }
}

EXPORT Matcher *bm_create(void) {return calloc(1,sizeof(Matcher));}
EXPORT void bm_free(Matcher *s) {free(s);}
EXPORT void bm_pair(Matcher *s,const uint8_t *current,const uint8_t *next,float *result) {
    if(!s->ready) thumbnail_pyramid(current,&s->frames[s->slot]);
    thumbnail_pyramid(next,&s->frames[1-s->slot]);
    memcpy(s->previous,s->latest,sizeof(s->latest));
    pair_flow(&s->frames[s->slot],&s->frames[1-s->slot],s->latest);
    if(!s->ready) memcpy(s->previous,s->latest,sizeof(s->latest));
    s->ready=1; s->slot=1-s->slot;
    if(result) {
        memcpy(result,s->previous,sizeof(s->previous));
        memcpy(result+2*PAIRS,s->latest,sizeof(s->latest));
    }
}

/* A piecewise-constant horizontal field preserves contiguous memory access. At each
   stripe centre the field is linearly interpolated from the six measured velocities.
   Along rows the field is continuous. */
typedef struct {int left,right; float top[2],change[2],extra[4][2];} Stripe;
typedef struct {float left,right,feature,threshold,value[2];} TreeNode;
typedef struct {
    const uint8_t *src;
    uint8_t *dst;
    const Stripe *layout;
    int first,last,stripes,sampler,dense;
} BlockJob;
enum { MAX_WORKERS=11 };
typedef struct {
    Matcher motion;
    int workers, stripes,sampler,queued,dense;
    float extra[4][6][2];
    BlockJob jobs[MAX_WORKERS+1];
    PTP_WORK work[MAX_WORKERS];
    PTP_WORK lookahead;
    const uint8_t *look_current,*look_next;
    Stripe layout[32];
    float weights[CELLS][2][FEATURES+1];
    const TreeNode *nodes;
    const int32_t *roots;
    int tree_count;
    float forest_bias[2], forest_gain;
} BlockRestorer;

static int mix_edge(const uint8_t *a,const uint8_t *b,int pos,int channel,const int w[4]) {
    int left=3*clamp_index(pos,1024)+channel,right=3*clamp_index(pos+1,1024)+channel;
    return (w[0]*a[left]+w[1]*a[right]+w[2]*b[left]+w[3]*b[right]+32)>>6;
}

static void mix_stripe(const uint8_t *a,const uint8_t *b,uint8_t *out,
                       int left,int right,int dx,int wx,int wy) {
    int w[4];
    w[1]=((64-wy)*wx+32)>>6; w[0]=64-wy-w[1];
    w[3]=(wy*wx+32)>>6; w[2]=wy-w[3];
    int first=left, last=right;
    if(first < -dx) first=-dx;
    if(first>right) first=right;
    if(last>1023-dx) last=1023-dx;
    if(last<first) last=first;
    for(int x=left;x<first;++x)
        for(int k=0;k<3;++k) out[3*x+k]=(uint8_t)mix_edge(a,b,x+dx,k,w);
    __m256i top=_mm256_set1_epi16((short)(w[0]+256*w[1]));
    __m256i bot=_mm256_set1_epi16((short)(w[2]+256*w[3]));
    __m256i round=_mm256_set1_epi16(32);
    int i=3*first,end=3*last;
    for(;i+32<=end;i+=32) {
        int j=i+3*dx;
        __m256i p=_mm256_loadu_si256((const __m256i*)(a+j)),q=_mm256_loadu_si256((const __m256i*)(a+j+3));
        __m256i r=_mm256_loadu_si256((const __m256i*)(b+j)),s=_mm256_loadu_si256((const __m256i*)(b+j+3));
        __m256i lo=_mm256_add_epi16(round,_mm256_add_epi16(_mm256_maddubs_epi16(_mm256_unpacklo_epi8(p,q),top),
                                                           _mm256_maddubs_epi16(_mm256_unpacklo_epi8(r,s),bot)));
        __m256i hi=_mm256_add_epi16(round,_mm256_add_epi16(_mm256_maddubs_epi16(_mm256_unpackhi_epi8(p,q),top),
                                                           _mm256_maddubs_epi16(_mm256_unpackhi_epi8(r,s),bot)));
        _mm256_storeu_si256((__m256i*)(out+i),_mm256_packus_epi16(_mm256_srli_epi16(lo,6),_mm256_srli_epi16(hi,6)));
    }
    for(;i<end;++i) {
        int j=i+3*dx;
        out[i]=(uint8_t)((w[0]*a[j]+w[1]*a[j+3]+w[2]*b[j]+w[3]*b[j+3]+32)>>6);
    }
    for(int x=last;x<right;++x)
        for(int k=0;k<3;++k) out[3*x+k]=(uint8_t)mix_edge(a,b,x+dx,k,w);
}

static void two_tap(const uint8_t *src,uint8_t *out,int left,int right,int y,int dx,int dy,
                    float rx,float ry) {
    int xs=rx<0?-1:1, ys=ry<0?-1:1;
    float ax=fabsf(rx),ay=fabsf(ry), weight;
    if(ay<.414214f*ax) {ys=0;weight=ax;}
    else if(ax<.414214f*ay) {xs=0;weight=ay;}
    else weight=(ax+ay)*.5f;
    const uint8_t *a=src+3072*clamp_index(y+dy,768),*b=src+3072*clamp_index(y+dy+ys,768);
    int wa=64-(int)(64*weight+.5f),wb=64-wa;
    const int first=(int)fminf(right,fmaxf(left,-dx+(xs<0)));
    const int last=(int)fmaxf(first,fminf(right,1024-dx-(xs>0)));
    for(int x=left;x<first;++x) for(int k=0;k<3;++k)
        out[3*x+k]=(uint8_t)((wa*a[3*clamp_index(x+dx,1024)+k]+wb*b[3*clamp_index(x+dx+xs,1024)+k]+32)>>6);
    __m256i w=_mm256_set1_epi16((short)(wa+256*wb)),round=_mm256_set1_epi16(32);
    int i=3*first,end=3*last;
    for(;i+32<=end;i+=32) {
        __m256i v=_mm256_loadu_si256((const __m256i*)(a+i+3*dx));
        __m256i u=_mm256_loadu_si256((const __m256i*)(b+i+3*(dx+xs)));
        __m256i lo=_mm256_add_epi16(round,_mm256_maddubs_epi16(_mm256_unpacklo_epi8(v,u),w));
        __m256i hi=_mm256_add_epi16(round,_mm256_maddubs_epi16(_mm256_unpackhi_epi8(v,u),w));
        _mm256_storeu_si256((__m256i*)(out+i),_mm256_packus_epi16(_mm256_srli_epi16(lo,6),_mm256_srli_epi16(hi,6)));
    }
    for(;i<end;++i) out[i]=(uint8_t)((wa*a[i+3*dx]+wb*b[i+3*(dx+xs)]+32)>>6);
    for(int x=last;x<right;++x) for(int k=0;k<3;++k)
        out[3*x+k]=(uint8_t)((wa*a[3*clamp_index(x+dx,1024)+k]+wb*b[3*clamp_index(x+dx+xs,1024)+k]+32)>>6);
}

static void block_rows(BlockJob *j) {
    for(int y=j->first;y<j->last;++y) {
        float vertical=fminf(1.4f,fmaxf(-.4f,(y-232.f)/312.f));
        float row=y-383.5f;
        float v=fminf(3.f,fmaxf(0.f,(y+.5f)/192.f-.5f));int level=(int)v;
        if(level>2)level=2;
        v-=level;
        for(int t=0;t<j->stripes;++t) {
            const Stripe *stripe=&j->layout[t];
            int x0=stripe->left,x1=stripe->right;
            float xf=row*(stripe->top[0]+vertical*stripe->change[0]);
            float offset=row*(stripe->top[1]+vertical*stripe->change[1]);
            if(j->dense) {
                xf+=row*(stripe->extra[level][0]+v*(stripe->extra[level+1][0]-stripe->extra[level][0]));
                offset+=row*(stripe->extra[level][1]+v*(stripe->extra[level+1][1]-stripe->extra[level][1]));
            }
            if(j->sampler==1) {
                int dx=(int)floorf(xf+.5f),dy=(int)floorf(offset+.5f);
                two_tap(j->src,j->dst+y*3072,x0,x1,y,dx,dy,xf-dx,offset-dy);
                continue;
            }
            float yf=y+offset;
            int dx=(int)floorf(xf),dy=(int)floorf(yf);
            int wx=(int)((xf-dx)*64+.5f),wy=(int)((yf-dy)*64+.5f);
            const uint8_t *a=j->src+3072*clamp_index(dy,768), *b=j->src+3072*clamp_index(dy+1,768);
            mix_stripe(a,b,j->dst+3072*y,x0,x1,dx,wx,wy);
        }
    }
}

static void CALLBACK block_callback(PTP_CALLBACK_INSTANCE instance,void *context,PTP_WORK work) {
    (void)instance;(void)work;block_rows(context);
}
static void CALLBACK estimate_ahead(PTP_CALLBACK_INSTANCE instance,void *context,PTP_WORK work) {
    (void)instance;(void)work;
    BlockRestorer *s=context;
    bm_pair(&s->motion,s->look_current,s->look_next,NULL);
}
EXPORT void block_delete(BlockRestorer *s) {
    if(!s) return;
    if(s->lookahead) CloseThreadpoolWork(s->lookahead);
    for(int k=0;k<s->workers;++k) if(s->work[k]) CloseThreadpoolWork(s->work[k]);
    free(s);
}
EXPORT BlockRestorer *block_new(int threads,int stripes,const float *weights) {
    if(threads<1 || threads>MAX_WORKERS+1 || stripes<1 || stripes>32) return NULL;
    BlockRestorer *s=calloc(1,sizeof(*s));
    if(!s) return NULL;
    s->workers=threads-1; s->stripes=stripes;
    memcpy(s->weights,weights,sizeof(s->weights));
    s->lookahead=CreateThreadpoolWork(estimate_ahead,s,NULL);
    if(!s->lookahead) {block_delete(s);return NULL;}
    for(int k=0;k<s->workers;++k) {
        s->work[k]=CreateThreadpoolWork(block_callback,&s->jobs[k],NULL);
        if(!s->work[k]) {block_delete(s);return NULL;}
    }
    return s;
}
EXPORT void block_sampler(BlockRestorer *s,int sampler) {s->sampler=sampler;}
/* The caller owns the fitted arrays for the lifetime of a sequence. */
EXPORT void block_trees(BlockRestorer *s,const TreeNode *nodes,const int32_t *roots,int count,const float *bias,float gain) {
    s->nodes=nodes;s->roots=roots;s->tree_count=count;s->forest_gain=gain;
    memcpy(s->forest_bias,bias,sizeof(s->forest_bias));
}
EXPORT void block_render(BlockRestorer *s,const uint8_t *image,uint8_t *out,const float *velocity) {
    float slope[CELLS][2];
    for(int p=0;p<CELLS;++p) {
        float f=3.6f/(767.f+3.6f*velocity[p*2+1]);
        for(int d=0;d<2;++d) slope[p][d]=f*velocity[2*p+d];
    }
    for(int t=0;t<s->stripes;++t) {
        Stripe *z=&s->layout[t];
        z->left=1024*t/s->stripes;z->right=1024*(t+1)/s->stripes;
        float u=fminf(2.4f,fmaxf(-.4f,((z->left+z->right)*.5f-224.f)/288.f));
        int c=u>1.f?1:0;u-=c;
        for(int d=0;d<2;++d) {
            z->top[d]=slope[c][d]+u*(slope[c+1][d]-slope[c][d]);
            float bottom=slope[c+3][d]+u*(slope[c+4][d]-slope[c+3][d]);
            z->change[d]=bottom-z->top[d];
        }
        if(s->dense) {
            float gx=fminf(5.f,fmaxf(0.f,((z->left+z->right)*.5f)*(6.f/1024.f)-.5f));
            int ix=(int)gx;if(ix>4)ix=4;gx-=ix;
            for(int iy=0;iy<4;++iy)for(int d=0;d<2;++d)
                z->extra[iy][d]=(3.6f/767.f)*(s->extra[iy][ix][d]+gx*(s->extra[iy][ix+1][d]-s->extra[iy][ix][d]));
        }
    }
    int count=s->workers+1;
    for(int k=0;k<count;++k) {
        BlockJob *j=&s->jobs[k];
        j->src=image; j->dst=out; j->first=768*k/count; j->last=768*(k+1)/count; j->stripes=s->stripes;
        j->layout=s->layout;j->sampler=s->sampler;j->dense=s->dense;
        if(k<s->workers) SubmitThreadpoolWork(s->work[k]);
        else block_rows(j);
    }
    for(int k=0;k<s->workers;++k) WaitForThreadpoolWorkCallbacks(s->work[k],FALSE);
}
static void block_predict(BlockRestorer *s,float *features,float *velocity) {
    features[FEATURES]=1;
    for(int i=0;i<CELLS*2;++i) {
        const float *w=s->weights[i/2][i%2];
        float sum=0;
        for(int k=0;k<=FEATURES;++k) sum+=w[k]*features[k];
        velocity[i]=sum;
    }
    if(s->tree_count) {
        float common[2][2]={{0}};
        for(int t=0;t<2;++t) for(int p=0;p<6;++p) for(int d=0;d<2;++d)
            common[t][d]+=features[t*14+2*(1+p)+d]/6.f;
        for(int p=0;p<6;++p) {
            float f[16],result[2]={s->forest_bias[0],s->forest_bias[1]};
            for(int d=0;d<2;++d) f[d]=velocity[p*2+d];
            f[2]=p%3-1;f[3]=2*(p/3)-1;
            for(int t=0;t<2;++t) for(int d=0;d<2;++d) {
                f[4+6*t+d]=features[14*t+2*(1+p)+d]-f[d];
                f[6+6*t+d]=features[14*t+d]-f[d];
                f[8+6*t+d]=common[t][d]-f[d];
            }
            for(int i=0;i<s->tree_count;++i) {
                const TreeNode *node=s->nodes+s->roots[i];
                while(node->left>=0) {
                    int next=(int)(f[(int)node->feature]<=node->threshold?node->left:node->right);
                    node=s->nodes+next;
                }
                result[0]+=node->value[0];result[1]+=node->value[1];
            }
            for(int d=0;d<2;++d) velocity[2*p+d]=(f[d]+result[d])*s->forest_gain;
        }
    }
}
EXPORT void block_correct(BlockRestorer *s,const uint8_t *current,const uint8_t *next,uint8_t *out) {
    float features[FEATURES+1],velocity[CELLS*2];
    bm_pair(&s->motion,current,next,features);
    block_predict(s,features,velocity);block_render(s,current,out,velocity);
}
EXPORT void block_stream(BlockRestorer *s,const uint8_t *current,const uint8_t *next,const uint8_t *future,uint8_t *out) {
    float features[FEATURES+1],velocity[CELLS*2];
    if(!s->queued) bm_pair(&s->motion,current,next,features);
    else {
        memcpy(features,s->motion.previous,sizeof(s->motion.previous));
        memcpy(features+2*PAIRS,s->motion.latest,sizeof(s->motion.latest));
    }
    s->queued=future!=NULL;
    if(future) {
        s->look_current=next;s->look_next=future;
        SubmitThreadpoolWork(s->lookahead);
    }
    block_predict(s,features,velocity);block_render(s,current,out,velocity);
    if(future) WaitForThreadpoolWorkCallbacks(s->lookahead,FALSE);
}
