function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--startidx"
            help = "Start index"
            arg_type = Int
            default = 1
        "--endidx"
            help = "End index"
            arg_type = Int
            default = 1040
        "--n_offsets"
            help = "num of offsets"
            arg_type = Int
            default = 51
        "--offset_start"
            help = "start of offset"
            arg_type = Float32
            default = -500f0
        "--offset_end"
            help = "end of offset"
            arg_type = Float32
            default = 500f0
        "--keep_offset_num"
            help = "keep how many offset during training"
            arg_type = Int
            default = 51
        "--free_surface"
            help = "free surface or absorbing boundary"
            action = :store_true
        "--with_density"
            help = "data modeled with density"
            action = :store_true
        "--resample_x"
            help = "resample x particles during WISER"
            action = :store_true
        "--redraw"
            help = "redraw white noise z during WISER"
            action = :store_true
        "--fc"
            help = "frequency continuation scheme"
            action = :store_true
        "--with_salt"
            help = "OOD with salt bodies"
            action = :store_true
        "--overthrust"
            help = "OOD using overthrust model"
            action = :store_true
        "--amplitude"
            help = "OOD amplitude"
            arg_type = Float32
            default = 0f0
        "--test_snr"
            help = "snr of noise during inference"
            arg_type = Float32
            default = 12f0
        "--wiser_batch_size"
            help = "number of particles during WISER"
            arg_type = Int
            default = 8
        "--lambda"
            help = "lambda in WISER"
            arg_type = Float32
            default = 0f0
        "--gamma"
            help = "gamma in WISER"
            arg_type = Float32
            default = 0f0
        "--unwiser"
            help = "pure band-limited noise in y"
            action = :store_true
        "--trend"
            help = "add curtis trend to compass model"
            action = :store_true
        "--fwi_only"
            help = "only do FWI, not WISER"
            action = :store_true
        "--freq1"
            help = "frequency continuation, 1st frequency"
            arg_type = Float32
            default = 4f0
        "--freq2"
            help = "frequency continuation, 2nd frequency"
            arg_type = Float32
            default = 6f0
        "--freq3"
            help = "frequency continuation, 3rdd frequency"
            arg_type = Float32
            default = 10f0
        "--iter1"
            help = "frequency continuation, 1st iter num"
            arg_type = Int
            default = 50         
        "--iter2"
            help = "frequency continuation, 2nd iter num"
            arg_type = Int
            default = 80
        "--iter3"
            help = "frequency continuation, 3rd iter num"
            arg_type = Int
            default = 120  
        "--inner_loop"
            help = "inner loops for weak WISER"
            arg_type = Int
            default = 128
        "--lr_wiser"
            help = "learning rate in WISER"
            arg_type = Float32
            default = 0.0004f0
    end
    return parse_args(s)
end

function de_z_shape_simple(G::NetworkGlow, X::AbstractArray{T, N}) where {T, N}
    G.split_scales && (Z_save = array_of_array(X, max(G.L-1,1)))

    logdet_ = 0
    for i=1:G.L
        (G.split_scales) && (X = G.squeezer.forward(X))
        if G.split_scales && (i < G.L || i == 1)    # don't split after last iteration
            X, Z = tensor_split(X)
            Z_save[i] = Z
            G.Z_dims[i] = collect(size(Z))
        end
    end
    G.split_scales && (X = cat_states(Z_save, X))

    return X
end

function z_shape_simple(G::NetworkGlow, ZX_test::AbstractArray{T, N}) where {T, N}
    Z_save, ZX = split_states(ZX_test[:], G.Z_dims)
    for i=G.L:-1:1
        if i < G.L
            ZX = tensor_cat(ZX, Z_save[i])
        end
        ZX = G.squeezer.inverse(ZX) 
    end
    ZX
end

function v_to_rho(vp::Matrix{T}, wb) where T
    rho = 0.31f0 .* (vp .* 1f3).^0.25f0
    rho[:, 1:wb-1] .= 1f0
    return rho
end

function v_to_rho(vp::Array{T, 4}, wb) where T
    rho = 0.31f0 .* (vp .* 1f3).^0.25f0
    rho[:, 1:wb-1, :, :] .= 1f0
    return rho
end

function ContJitter(l::Number, num::Int)
    #l = length, num = number of samples
    interval_width = l/num
    interval_center = range(interval_width/2, stop = l-interval_width/2, length=num)
    randomshift = interval_width .* rand(Float32, num) .- interval_width/2

    return interval_center .+ randomshift
end