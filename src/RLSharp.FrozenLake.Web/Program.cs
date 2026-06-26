using RLSharp.FrozenLake.Web.Hub;
using RLSharp.FrozenLake.Web.Services;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddSignalR();
builder.Services.AddSingleton<ITrainingService, TrainingService>();
builder.Services.AddCors(o => o.AddDefaultPolicy(p => p
	.AllowAnyHeader().AllowAnyMethod().AllowCredentials()
	.SetIsOriginAllowed(_ => true)));

var app = builder.Build();

app.UseDefaultFiles();
app.UseStaticFiles();
app.UseCors();
app.MapHub<TrainingHub>("/trainingHub");

app.Run();