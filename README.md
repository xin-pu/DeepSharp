# ReinforcementLearning
Secondary development by torchsharp for Deep Learning and Reinforcement Learning

## DataSet 

Todo

``` C#
public class IrisData : DataView
{
    /// <summary>
    /// </summary>
    public IrisData()
    {
    }

    [StreamHeader(0)] public float Label { set; get; }
    [StreamHeader(1)] public float SepalLength { set; get; }
    [StreamHeader(2)] public float SepalWidth { set; get; }
    [StreamHeader(3)] public float PetalLength { set; get; }
    [StreamHeader(4)] public float PetalWidth { set; get; }

    public override torch.Tensor GetFeatures()
    {
        return torch.tensor(new[] {SepalLength, SepalWidth, PetalLength, PetalWidth});
    }

    public override torch.Tensor GetLabels()
    {
        return torch.tensor(new[] {Label});
    }

      
}
```


``` C#
var dataset = new Dataset<IrisData>(@"F:\Iris\iris-train.txt");
var res = dataset.GetTensor(0);
Print(res);
```


## DataLoader

``` C#
var dataset = new Dataset<IrisData>(@"F:\Iris\iris-train.txt");
var dataConfig = new DataLoaderConfig
{
    Device = new torch.Device(DeviceType.CUDA)
};
var dataloader = new DataLoader<IrisData>(dataset, dataConfig);

using var iterator = dataloader.GetEnumerator();
while (iterator.MoveNext())
{
    var current = iterator.Current;
    Print(current);
}
```

## InfiniteDataLoader

```c#

var dataset = new Dataset<IrisData>(@"F:\Iris\iris-train.txt");
var dataConfig = new DataLoaderConfig();
var dataloader = new InfiniteDataLoader<IrisData>(dataset, dataConfig);

await foreach (var a in dataloader.GetBatchSample(100))
{
    var array = a.Labels.data<float>().ToArray();
    Print($"{string.Join(";", array)}");
}
```


## RL

Slove KArmedBandit Problem by Cross Entropy Deep Reinforcement Learning

1. Develop  KArmedBandit which Inherit  from Environ
2. Create KArmedBandit with 4 Bandit
3. Create AgentCrossEntropy by send KArmedBandit
4. Learn by Epoch
    1. Get Batch Episodes
    2. Filter Elite from Batch Episodes
    3. Agent learn by  Elite
    4. Next epoch until exit learn.


``` c#
var epoch = 100;
var episodesEachBatch = 20;

/// Step 1 Create a 4-Armed Bandit
var kArmedBandit = new KArmedBandit(4);
Print(kArmedBandit);

/// Step 2 Create AgentCrossEntropy with 0.7f percentElite as default
var agent = new AgentCrossEntropy(kArmedBandit);

/// Step 3 Learn and Optimize
foreach (var i in Enumerable.Range(0, epoch))
{
    var batch = kArmedBandit.GetMultiEpisodes(agent, episodesEachBatch);
    var eliteOars = agent.GetElite(batch); /// Get eliteOars 

    /// Agent Learn by elite observation & action
    var loss = agent.Learn(eliteOars);
    var rewardMean = batch.Select(a => a.SumReward.Value).Average();

    Print($"Epoch:{i:D4}\tReward:{rewardMean:F4}\tLoss:{loss:F4}");
}
```

![R L Cross Entroy Demo](images/RL%20CrossEntroy%20Demo.png)
