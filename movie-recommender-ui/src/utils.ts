export const loadJsonData = (j: Object): string => JSON.parse(JSON.stringify(j));

export const sorterStringCompare = (str1: string, str2: string) => {
    return (str1 || '').localeCompare(str2 || '');
}