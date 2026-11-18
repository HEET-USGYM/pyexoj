// Run ExoJ headlessly via ExoJCli. Argument: path to a .properties settings file.
settings = getArgument();
if (settings == "") exit("usage: fiji -batch exoj_run.ijm /path/settings.properties");
r = call("ExoJCli.run", settings);
print("ExoJCli: " + r);
