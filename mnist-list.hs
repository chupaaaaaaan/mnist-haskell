-- {-# LANGUAGE Strict, StrictData, GADTs #-}
{-# LANGUAGE GADTs #-}

import Control.Monad
import Control.Exception
import Numeric.LinearAlgebra
import qualified Data.ByteString.Lazy as B
import Data.Maybe
import Data.Binary
import Data.IORef
import System.IO
import System.Random

import Function

import Debug.Trace

-- definition ----------------------------------------------------
type Image = Vector R
type Label = Vector R
type DataSet = [(Image, Label)]
type W2 = Matrix R
type B2 = Vector R
type W3 = Matrix R
type B3 = Vector R
type ParamSet = (W2, B2, W3, B3)

data SType = Train | Shuffle | Test deriving (Eq)

-- constants ----------------------------------------------------
dataSet :: [(SType,(FilePath,FilePath))]
dataSet =
  [(Shuffle, ("/vagrant/train-images.idx3-ubyte-shuffle", "/vagrant/train-labels.idx1-ubyte-shuffle"))
  ,(Train,   ("/vagrant/train-images.idx3-ubyte",         "/vagrant/train-labels.idx1-ubyte"))
  ,(Test,    ("/vagrant/t10k-images.idx3-ubyte",          "/vagrant/t10k-labels.idx1-ubyte"))
  ]

trainDataNum :: Int
trainDataNum = 60000

testDataNum :: Int
testDataNum = 10000

bsize :: Int -- mini batch size
bsize = 100

epochUnit :: Int
epochUnit = (trainDataNum + bsize -1) `div` bsize

trainEpoch :: Int
trainEpoch = 16

trainNum :: Int
trainNum = epochUnit * trainEpoch

imgSize :: Int
imgSize = 784

ep :: R -- learning rate
-- ep = 1.0e-2
ep = 1.0e-1

mu :: R -- momentum rate
-- mu = 0.9
mu = 0.0



-- utilities for loading data ----------------------------------------------------
toDataSet :: SType -> IO DataSet
toDataSet s = do
  let imgpath = maybe "" fst . lookup s $ dataSet
      lblpath = maybe "" snd . lookup s $ dataSet
  imgData <- loadImg imgpath
  lblData <- loadLbl lblpath
  return $ zip imgData lblData

loadImg :: FilePath -> IO [Vector R]
loadImg fp = do
  contents <- B.readFile fp
  let doubleList = toDoubleList $ B.drop 16 contents
      vecList    = slice2vec imgSize doubleList
  return vecList

loadLbl :: FilePath -> IO [Vector R]
loadLbl fp = do
  contents <- B.readFile fp
  let doubleList = toOneHotList $ B.drop 8 contents
      vecList    = slice2vec 10 doubleList
  return vecList

slice :: ([a] -> b) -> Int -> [a] -> [b]
slice _ _ [] = []
slice f n xs = f (take n xs) : slice f n (drop n xs)

slice2list :: Int -> [a] -> [[a]]
slice2list = slice id

slice2vec :: Int -> [R] -> [Vector R]
slice2vec = slice vector

onehot :: (Integral a) => a -> a -> [a]
onehot m n = reverse $ oh m
  where
    oh 0 = []
    oh x = (if n == (x - 1) then 1 else 0) : oh (x - 1)

toDoubleList :: B.ByteString -> [R]
toDoubleList = map ((/255) . read . show . fromEnum) . B.unpack

toOneHotList :: B.ByteString -> [R]
toOneHotList = concatMap (map fromIntegral . onehot 10 . fromEnum) . B.unpack


-- utility for initializing weights ----------------------------------------------------
initMatrix :: Int -> Int -> IO (Matrix R)
initMatrix row col = (scalar 0.01 *) <$> randn row col

initVector :: Int -> IO (Vector R)
initVector len = ((scalar 0.01 *) . flatten) <$> randn 1 len


-- elements of learning network ----------------------------------------------------
activate :: R -> R
activate = relu

activate' :: R -> R
activate' = relu'

forward :: ParamSet -> Vector R -> Vector R
forward (w2,b2,w3,b3) input = w3 #> cmap activate (w2 #> input + b2) + b3

accuracy_count :: ParamSet -> DataSet -> R
accuracy_count param = (sum . map accept)
  where accept (img,lbl) = if maxIndex (forward param img) == maxIndex lbl then 1.0 else 0.0

loss :: ParamSet -> DataSet -> R
loss param = (/ fromIntegral bsize) . sum . map cross_entropy
  where cross_entropy (img,lbl) = - (lbl <.> cmap log (scalar 1e-7 + softmax (forward param img)))

update :: ParamSet -> ParamSet -> ParamSet -> ParamSet
update original gradient momentum = original `sumParam` gradient `sumParam` momentum

grad :: ParamSet -> DataSet -> ParamSet
grad (w2,b2,w3,b3) = foldr (sumParam . getParamDiff) acc
  where acc = (0.0,0.0,0.0,0.0)
        getParamDiff (img,lbl) =
          -- forward propagation
          let u2 = w2 #> img + b2
              z2 = cmap activate u2
              u3 = w3 #> z2 + b3
              z3 = softmax u3
          -- back propagation
              d3 = z3 - lbl
              d2 = cmap activate' u2 * (d3 <# w3)
              dw3 = d3 `outer` z2
              db3 = d3
              dw2 = d2 `outer` img
              db2 = d2
          in (dw2,db2,dw3,db3)

sumParam :: ParamSet -> ParamSet -> ParamSet
sumParam (a0,b0,c0,d0) (a1,b1,c1,d1) = (a0+a1,b0+b1,c0+c1,d0+d1)

mulParam :: R -> ParamSet -> ParamSet
mulParam x (a0,b0,c0,d0) =
  (scalar x * a0, scalar x * b0, scalar x * c0, scalar x * d0)

finiteRandomRs :: (RandomGen g, Random a, Num n, Eq n) => (a,a) -> n -> g -> ([a], g)
finiteRandomRs _ 0 gen = ([], gen)
finiteRandomRs (x,y) n gen =
  let (value, newGen) = randomR (x,y) gen
      (restOfList, finalGen) = finiteRandomRs (x,y) (n-1) newGen
  in (value:restOfList, finalGen)


-- main ----------------------------------------------------
main :: IO ()
main = do
  -- initialize
  w2 <- initMatrix 50 imgSize
  b2 <- (scalar 0 *) <$> initVector 50
  w3 <- initMatrix 10 50
  b3 <- (scalar 0 *) <$> initVector 10

  -- variables
  refp <- newIORef (w2, b2, w3, b3)
  refm <- newIORef (0.0, 0.0, 0.0, 0.0)
  -- refac_train <- newIORef 0
  -- refac_test <- newIORef 0

  -- fetch dataset
  shuffles <- toDataSet Shuffle
  -- trains <- toDataSet Train
  -- tests <- toDataSet Test

  -- split to mini batch
  let mbs = slice2list bsize shuffles
      loop = zip (take trainNum [0..]) mbs
      -- train_mbs = slice2list bsize trains
      -- test_mbs = slice2list bsize trains

  -- learning steps
  forM_ loop $ \(i,mb) -> do
    p <- readIORef refp
    m <- readIORef refm

    -- calculate gradient and momentum
    let gradient = (-ep / fromIntegral bsize) `mulParam` grad p mb
        momentum = mu `mulParam` m

    -- update paramSet
    writeIORef refp (update p gradient momentum)

    -- set next momentum
    writeIORef refm gradient

    -- output error
    p' <- readIORef refp
    let dataline = show i ++ " " ++ show (accuracy_count p' mb) ++ " " ++ show (loss p' mb)
    appendFile "error.dat" dataline
    putStrLn dataline

    -- -- output accuracy
    -- when ((i+1) `mod` epochUnit == 0) $ do
    --   writeIORef refac_train 0
    --   writeIORef refac_test 0
    --   forM_ train_mbs $ \mb -> modifyIORef refac_train (+ accuracy_count p' mb)
    --   forM_ test_mbs $ \mb -> modifyIORef refac_test (+ accuracy_count p' mb)
    --   ac_train <- readIORef refac_train
    --   ac_test <- readIORef refac_test
    --   print ac_train

    --   let dataline = show ((i+1) `div` epochUnit) ++ " " ++
    --                  show (ac_train / fromIntegral trainDataNum) ++ " " ++
    --                  show (ac_test / fromIntegral testDataNum)
    --   appendFile "accuracy.dat" dataline
    --   putStrLn dataline

  return ()
