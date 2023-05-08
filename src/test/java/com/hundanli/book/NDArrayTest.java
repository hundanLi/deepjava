package com.hundanli.book;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import org.junit.jupiter.api.Test;

/**
 * @author hundanli
 * @version 1.0.0
 * @date 2023/3/5 23:20
 */
public class NDArrayTest {

    @Test
    void create() throws Exception {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray ndArray = manager.arange(12);
            System.out.println(ndArray);
            ndArray = ndArray.reshape(3, -1);
            System.out.println(ndArray);

            ndArray = manager.create(new Shape(2, 3));
            System.out.println(ndArray);

            ndArray = manager.zeros(new Shape(2, 3));
            System.out.println(ndArray);

            ndArray = manager.randomNormal(new Shape(2, 3));
            System.out.println(ndArray);
        }

    }


    @Test
    void operator() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray x = manager.create(new float[]{1F, 2F, 4F, 8F});
            NDArray y = manager.create(new float[]{2F, 4F, 8F, 16F});
            NDArray add = x.add(y);
            System.out.println("add:" + add);
            NDArray sub = x.sub(y);
            System.out.println("sub:" + sub);
            NDArray mul = x.mul(y);
            System.out.println("mul:" + mul);
            NDArray div = x.div(y);
            System.out.println("div:" + div);
            NDArray pow = x.pow(y);
            System.out.println("pow:" + pow);
            NDArray exp = x.exp();
            System.out.println("exp:" + exp);
            x = manager.arange(12F).reshape(3, -1);
            y = manager.create(new float[]{2, 1, 4, 3, 1, 2, 3, 4, 4, 3, 2, 1}).reshape(3, -1);
            NDArray concat = x.concat(y);// axis=0
            System.out.println("axis=0,concat:" + concat);
            concat = x.concat(y, 1);
            System.out.println("axis=1,concat:" + concat);
            NDArray eq = x.eq(y);
            System.out.println("eq:"+eq);
            NDArray sum = x.sum();
            System.out.println("sum:"+sum);

            // 原地操作，节省内存开销
            NDArray actual = x.addi(1);
            System.out.println(actual == x);

        }
    }

}
