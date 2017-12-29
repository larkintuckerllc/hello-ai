import {
  CostReduction,
  ENV,
  Graph,
  InCPUMemoryShuffledInputProviderBuilder,
  Scalar,
  Session,
  SGDOptimizer,
} from 'deeplearn';

const LEARNING_RATE = .1;
const BATCH_SIZE = 1;
const NUM_BATCHES = 100;
const math = ENV.math;
const g = new Graph();
const xP = g.placeholder('xP', []);
const yP = g.placeholder('yP', []);
const mV = g.variable('mV', Scalar.new(0.0));
const bV = g.variable('bV', Scalar.new(0.0));
const y = g.add(g.multiply(mV, xP), bV);
const cost = g.reduceSum(g.square(g.subtract(y, yP)));
const session = new Session(g, math);
const optimizer = new SGDOptimizer(LEARNING_RATE);
const xs = [
  Scalar.new(0.0),
  Scalar.new(1.0),
  Scalar.new(2.0),
];
const ys = [
  Scalar.new(1.0),
  Scalar.new(3.0),
  Scalar.new(5.0),
];
const shuffledInputProviderBuilder =
  new InCPUMemoryShuffledInputProviderBuilder([xs, ys]);
const [xProvider, yProvider] =
  shuffledInputProviderBuilder.getInputProviders();
const feedEntries: FeedEntry[] = [
  {tensor: xP, data: xProvider},
  {tensor: yP, data: yProvider}
];
const batches = [];
for (let i = 0; i < NUM_BATCHES; i += 1) {
  const batch = session.train(
    cost,
    feedEntries,
    BATCH_SIZE,
    optimizer,
    CostReduction.MEAN,
  );
  batches.push(batch.data());
}
Promise.all(batches).then(() => {
  const evalX = Scalar.new(3.0);
  const evalFeedEntries: FeedEntry[] = [
    {tensor: xP, data: evalX}
  ];
  session.eval(y, evalFeedEntries).data()
    .then(data => {
      console.log(data);
    });
});
