export const loadJsonData = (j: Object): string => JSON.parse(JSON.stringify(j));

export const sorterStringCompare = (str1: string, str2: string) => {
    return (str1 || '').localeCompare(str2 || '');
}

// Generator function to split an array into chunks
export function* chunks(arr: Object[], n: number) {
    for (let i = 0; i < arr.length; i += n) {
        yield arr.slice(i, i + n);
    }
}
