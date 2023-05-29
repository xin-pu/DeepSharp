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