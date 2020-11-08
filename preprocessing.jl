using DelimitedFiles
using Serialization
using SparseArrays


function load_lastfm(path="./lastfm-dataset-360K/usersha1-artmbid-artname-plays.tsv")
    function index!(dict, name)
        i = get(dict, name, 0)
        if i == 0
            dict[name] = i = length(dict) + 1
        end
        return i
    end

    data = Array{Int}(undef, countlines(open(path)), 3)
    users = Dict{String, Int}()
    items = Dict{String, Int}()
    for (i, line) in enumerate(eachline(path))
        cells = split(line, '\t')
        data[i, 1] = index!(users, cells[1])
        data[i, 2] = index!(items, cells[2])
        data[i, 3] = parse(Int, cells[4])
    end

    return data
end

lastfm = load_lastfm()
lastfm = sparse(view(lastfm, :, 2),
                view(lastfm, :, 1),
                view(lastfm, :, 3))
serialize("lastfm.jlser", lastfm)
