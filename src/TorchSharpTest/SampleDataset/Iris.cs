using DeepSharp.Dataset;

namespace TorchSharpTest.SampleDataset
{
	public class Iris : DataView
	{
		/// <summary>
		/// </summary>
		public Iris()
		{
		}

		[StreamHeader(0)]
		public long Label { get; set; }

		[StreamHeader(1)]
		public float SepalLength { get; set; }

		[StreamHeader(2)]
		public float SepalWidth { get; set; }

		[StreamHeader(3)]
		public float PetalLength { get; set; }

		[StreamHeader(4)]
		public float PetalWidth { get; set; }

		public override Tensor GetFeatures()
		{
			return tensor(new[] { SepalLength, SepalWidth, PetalLength, PetalWidth });
		}

		public override Tensor GetLabels()
		{
			return tensor(new[] { Label });
		}

		public override string ToString()
		{
			return $"Label:{Label}\t"                                            +
			       $"SepalLength:{SepalLength:F2}\tSepalWidth:{SepalWidth:F2}\t" +
			       $"PetalLength:{PetalLength:F2}\tPetalWidth:{PetalWidth:F2}";
		}

		/// <summary>
		///     return a random Iris
		/// </summary>
		/// <returns></returns>
		public static Iris RandomIris()
		{
			var randomSource = new Random();
			return new Iris
			{
				Label       = randomSource.Next(0, 3),
				PetalLength = randomSource.NextSingle() * 4,
				PetalWidth  = randomSource.NextSingle() * 4,
				SepalLength = randomSource.NextSingle() * 4,
				SepalWidth  = randomSource.NextSingle() * 4
			};
		}
	}
}