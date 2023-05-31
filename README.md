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

Slove KArmedBandit Problem by Cross Entropy Deep RFeinforcement Learning

``` c#
var k = 2;
var batchSize = 1000;
var percent = 0.7f;

/// Step 1 创建环境
var kArmedBandit = new KArmedBandit(k);
Print(kArmedBandit);

/// Step 2 创建智能体
var agent = new AgentKArmedBandit(k, k);

/// Step 3 边收集 边学习
foreach (var i in Enumerable.Range(0, 200))
{
    var batch = kArmedBandit.GetBatchs(agent);
    var oars = agent.GetElite(batch, percent);

    var observation = torch.vstack(oars.Select(a => a.Observation.Value).ToList());
    var action = torch.vstack(oars.Select(a => a.Action.Value).ToList()).squeeze(-1);

    var rewardMean = batch.Select(a => a.SumReward.Value).Average();
    var loss = agent.Learn(observation, action);

    Print($"Epoch:{i:D4}\tReward:{rewardMean:F4}\tLoss:{loss:F4}");
}
```

![R L Cross Entroy Demo](images/RL%20CrossEntroy%20Demo.png)