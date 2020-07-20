using LinearAlgebra, Test
using CUDA

import LinearAlgebra: BlasInt

m = 256
n = 512
devs = collect(devices())
if length(devs) == 1
    devs = vcat(devs, copy(devs))
end

CUSOLVER.cusolverMgDeviceSelect(CUSOLVER.mg_handle(), length(devs), devs)
@testset "mg_syevd!" begin
    @testset "element type $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
        A = rand(elty, m, m)
        A += A'
        hW = eigvals(Hermitian(A))
        hV = eigvecs(Hermitian(A))
        W, A = CUSOLVER.mg_syevd!('V','L',A, devs=devs)
        # compare
        @test W ≈ hW 
        @test A*diagm(0=>W)*A' ≈ hV*diagm(0=>hW)*hV'

        A = rand(elty, m, m)
        A += A'
        hW = eigvals(Hermitian(A))
        W = CUSOLVER.mg_syevd!('N','L',A, devs=devs)
        # compare
        @test W ≈ hW 

    end
end # elty

if CUDA.toolkit_version() >= v"11.0"
    @testset "mg_potrf!" begin
        @testset "element type $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
            A = rand(elty, m, m)
            A = A*A'
            hA = copy(A)
            A = CUSOLVER.mg_potrf!('L',A, devs=devs)
            LinearAlgebra.LAPACK.potrf!('L', hA)
            # compare
            @test A ≈ hA 
        end
    end # elty

    @testset "mg_potrf and mg_potri!" begin
        #@testset "element type $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
        @testset "element type $elty" for elty in [Float64, ComplexF64]
            A = rand(elty, m, m)
            A = A*A'
            hA = copy(A)
            LinearAlgebra.LAPACK.potrf!('L', hA)
            A = CUSOLVER.mg_potrf!('L',A, devs=devs)
            LinearAlgebra.LAPACK.potri!('L', hA)
            A = CUSOLVER.mg_potri!('L',A, devs=devs)
            # compare
            @test tril(A) ≈ tril(hA)
        end
    end # elty

    @testset "mg_potrf and mg_potrs!" begin
        #@testset "element type $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
        @testset "element type $elty" for elty in [Float64, ComplexF64]
            A = rand(elty, m, m)
            B = rand(elty, m, m)
            A = A*A'
            hA = copy(A)
            hB = copy(B)
            LinearAlgebra.LAPACK.potrf!('L', hA)
            LinearAlgebra.LAPACK.potrs!('L', hA, hB)
            A = CUSOLVER.mg_potrf!('L', A, devs=devs)
            B = CUSOLVER.mg_potrs!('L', A, B, devs=devs)
            # compare
            tol    = real(elty) == Float32 ? 1e-4 : 1e-6
            @test A ≈ hA 
            @test B ≈ hB rtol=tol
        end
    end # elty
end

if CUDA.toolkit_version() >= v"10.2"
    @testset "mg_getrf!" begin
        @testset "element type $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
            A      = rand(elty,m,m)
            h_A    = copy(A)
            A,ipiv = CUSOLVER.mg_getrf!(A, devs=devs)
            alu    = LinearAlgebra.LU(A, convert(Vector{BlasInt},ipiv), zero(BlasInt))
            @test h_A ≈ Array(alu)
        end
    end

    @testset "mg_getrs!" begin
        @testset "element type $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
            A      = rand(elty,m,m)
            h_A    = copy(A)
            alu    = lu(A, Val(false))
            B      = rand(elty, m, div(m,2))
            h_B    = copy(B)
            tol    = real(elty) == Float32 ? 1e-1 : 1e-6
            B      = CUSOLVER.mg_getrs!('N', alu.factors, alu.ipiv, B, devs=devs)
            @test B ≈ h_A\h_B  rtol=tol
        end
    end
end
