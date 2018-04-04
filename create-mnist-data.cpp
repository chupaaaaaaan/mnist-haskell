#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

const int imgSize = 28*28;
const int trainDataNum = 60000;
const int trainEpoch = 16;
const int trainDataTotal = trainEpoch * trainDataNum;


unsigned char ibuf_train[trainDataNum][imgSize];
unsigned char lbuf_train[trainDataNum];
unsigned char idat_train[trainDataTotal][imgSize];
unsigned char ldat_train[trainDataTotal];

char ibuf[16];
char lbuf[8];


void input();
void shuffle();
void output();


int main(int argc,char **argv)
{
  input();

  int sample=24321;
  int tample=800000;

  for(int i=0;i<28;++i){
    for(int j=0;j<28;++j){
      printf("%u ",ibuf_train[sample][i*28+j]);
    }
    printf("\n");
  }
  printf("%u\n",lbuf_train[sample]);

  shuffle();

  for(int i=0;i<28;++i){
    for(int j=0;j<28;++j){
      printf("%u ",idat_train[tample][i*28+j]);
    }
    printf("\n");
  }
  printf("%u\n",ldat_train[tample]);


  output();
    
  return 0;
}

/******************************************************************
      input and output
******************************************************************/

void input()
{
  char train_img[] = "/vagrant/train-images.idx3-ubyte";
  char train_lbl[] = "/vagrant/train-labels.idx1-ubyte";

  FILE* fp;

  // training data : image
  if((fp=fopen(train_img,"rb"))==NULL){
    printf("Error opening %s.\n",train_img);
    exit(1);
  }
  fread(ibuf, sizeof(ibuf),1,fp);
  fread(ibuf_train, sizeof(unsigned char),trainDataNum*imgSize,fp);
  fclose(fp);

  // training data : label
  if((fp=fopen(train_lbl,"rb"))==NULL){
    printf("Error opening %s.\n",train_lbl);
    exit(1);
  }
  fread(lbuf, sizeof(lbuf),1,fp);
  fread(lbuf_train, sizeof(unsigned char),trainDataNum,fp);
  fclose(fp);

}

void shuffle()
{
  srand(392);
  int idx=0;

  for(int i=0;i<trainDataTotal;++i){
    idx = rand() % trainDataNum;
    memcpy(&idat_train[i][0], &ibuf_train[idx][0], imgSize);
    ldat_train[i] = lbuf_train[idx];
  }  
}

void output()
{
  char train_img[] = "/vagrant/train-images.idx3-ubyte-shuffle";
  char train_lbl[] = "/vagrant/train-labels.idx1-ubyte-shuffle";

  FILE* fp;

  // training data : image
  if((fp=fopen(train_img,"wb"))==NULL){
    printf("Error opening %s.\n",train_img);
    exit(1);
  }
  fwrite(ibuf, sizeof(ibuf),1,fp);
  fwrite(idat_train, sizeof(unsigned char),trainDataTotal*imgSize,fp);
  fclose(fp);

  // training data : label
  if((fp=fopen(train_lbl,"wb"))==NULL){
    printf("Error opening %s.\n",train_lbl);
    exit(1);
  }
  fwrite(lbuf, sizeof(lbuf),1,fp);
  fwrite(ldat_train, sizeof(unsigned char),trainDataTotal,fp);
  fclose(fp);

}
