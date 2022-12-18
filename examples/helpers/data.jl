
function getdatapath()::String

    dataset_alias = "AI4DBiological-Systems_NMR_data" # don't use spaces or 'strange' symbols like commas, colons, etc.
    archive_file_name = "archive.tar" # the filename on the data repository that we download.
    url = "https://github.com/AI4DBiological-Systems/PublicNMRData/raw/main/archive.tar"
    # archive_file_name = "archive.zip"
    # url = "https://github.com/AI4DBiological-Systems/PublicNMRData/raw/main/archive.zip"

    try
        @datadep_str("$dataset_alias")
    catch
        register(DataDep("$dataset_alias",
            """
            Dataset: Public NMR research data
            Author: National Research Council of Canada
            License: [Creative Commons Attribution Non Commercial Share Alike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)
            Website: https://github.com/AI4DBiological-Systems/PublicNMRData

            description here.
            Citation: National Research Council of Canada, Public NMR research data


            """,
            url
        ));
    end

    #readdir(datadep"AI4DBiological-Systems NMR data") # have to manually type out the alias. Does not allow string variable substitution.
    local_dataset_archive_path = @datadep_str("$dataset_alias") # call the actual macro to allow string variable substitution.

    # extract archive, then delete. Do this only if archive file still exists.

    root_data_path = joinpath(local_dataset_archive_path, "contents")
    
    if isfile(joinpath(local_dataset_archive_path, archive_file_name))
        t = @task begin; ispath(root_data_path) || mkpath(root_data_path); end
        schedule(t); wait(t)
    
        t = @task begin; Tar.extract(joinpath(local_dataset_archive_path, archive_file_name), root_data_path); end
        schedule(t); wait(t)
        rm(joinpath(local_dataset_archive_path, archive_file_name)) # delete the archive file.
    end

    return root_data_path

    # # return root_data_path. however, this unpacks in the current working directory!
    # archive_file_path = joinpath(local_dataset_archive_path, archive_file_name)
    # if isfile(archive_file_path)
    #     DataDeps.unpack(archive_file_path)
    # end
    #return local_dataset_archive_path
end